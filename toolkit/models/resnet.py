import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from einops import rearrange, repeat
from toolkit.models.head import (CrossLevelHead, SelfRelationHeadV2)

__all__ = ["resnet18", "resnet34", "resnet50"]


def bchw2bhwc(x):
    h, w = x.shape[-2:]
    return rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)


class ResnetWrapper(ResNet):
    def __init__(self, name, *args, **kwargs):
        super(ResnetWrapper, self).__init__(*args, **kwargs)
        self.name = name
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        setattr(self, 'num_features', self.fc.in_features)
        self.fc = nn.Identity()

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]

        # multi-view index
        view_sizes = [inp.shape[-1] for inp in x]
        unique_sizes_count = torch.tensor(view_sizes).unique_consecutive(return_counts=True)[1]
        idx_crops = unique_sizes_count.cumsum(0)

        if hasattr(self, "use_multi_level") and not self.use_multi_level:
            output = {}
            output_cls, output_fea, num_patch = [], [], []
            output_fea_wo_reshape = []
            # Multi-view forward
            for start_idx, end_idx in zip([0] + idx_crops[:-1].tolist(), idx_crops.tolist()):
                out_cls, out_fea = self.forward_features(torch.cat(x[start_idx: end_idx]))

                # Concatenate the features across patches
                batch_size, num_fea, channel = out_fea.shape
                output_cls.append(out_cls)
                output_fea_wo_reshape.append(out_fea)
                output_fea.append(out_fea.reshape(batch_size * num_fea, channel))

                # Record batch size
                num_patch.append(num_fea)

                del out_cls, out_fea

            if hasattr(self, "head") and output_cls:
                output["head"] = getattr(self, 'head')(torch.cat(output_cls))

            if hasattr(self, "dense_head") and output_fea:
                output["dense_head"] = getattr(self, 'dense_head')(torch.cat(output_fea))

            if hasattr(self, "mix_head") and output_cls and output_fea and num_patch:
                output["mix_head"] = getattr(self, 'mix_head')(output_cls, output_fea, num_patch)

            if hasattr(self, "trans_head"):
                output["trans_head"] = getattr(self, 'trans_head')(output_fea_wo_reshape)

            output["output_fea"] = torch.cat(output_fea)
            output["num_patch"] = num_patch
            return output

        elif hasattr(self, "use_multi_level") and self.use_multi_level:
            num_patch = [[] for _ in range(4)]  # [layer1, layer2, ...]
            output_fea = [[] for _ in range(4)]  # [layer1, layer2, ...]

            for start_idx, end_idx in zip([0] + idx_crops[:-1].tolist(), idx_crops.tolist()):
                out = self.multi_level_forward_features_w_pool(torch.cat(x[start_idx: end_idx]))
                for i, o in enumerate(out):
                    batch_size, patch, channel = o.shape
                    num_patch[i].append(patch)
                    output_fea[i].append(o.reshape(batch_size * patch, channel))

            output = {}
            for i in range(len(output_fea)):
                output_fea[i] = torch.cat(output_fea[i], dim=0)
            if hasattr(self, "multi_head") and self.use_multi_level:
                output['multi_level'] = getattr(self, 'multi_head')(output_fea)

            output['num_patch'] = num_patch
            output['output_fea'] = output_fea
            return output

        elif hasattr(self, "use_corr") and self.use_corr:
            output = {}

            batch_size = x[0].shape[0]
            # if feat len is 1 only contain global view
            # if feat len is 2 contain global and local
            # Multi-View Forward
            for start_idx, end_idx in zip([0] + idx_crops[:-1].tolist(), idx_crops.tolist()):
                # [layer1, layer2, layer3, layer4]
                # layer x : [a,b,c,d]
                ml_feat = self.multi_level_forward_features(torch.cat(x[start_idx: end_idx]))
                chunk_size = end_idx - start_idx
                if hasattr(self, "corr_head"):
                    if isinstance(self.corr_head, (CrossLevelHead, SelfRelationHeadV2)):
                        layers_output = self.corr_head(ml_feat)
                        for layer_idx, layer_output in enumerate(layers_output):
                            if layer_idx not in output.keys():
                                output[layer_idx] = {}

                            out1, out2 = layer_output
                            out1 = out1.chunk(chunk_size)  # view1, view2
                            out2 = out2.chunk(chunk_size)
                            for view_idx, (m, n) in enumerate(zip(out1, out2)):
                                output[layer_idx][start_idx + view_idx] = [m, n]

                    else:
                        raise ValueError(f"Not support for this {type(self.corr_head)}")

            return output
        else:
            raise ValueError("Not Support this type forward")

    def forward_features(self, x):
        x_region = self.forward_feature_map(x)
        x = self.avgpool(x_region)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, bchw2bhwc(x_region)

    def forward_feature_map(self, x):
        x = self.forward_stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def multi_level_forward_features_w_pool(self, x):
        x = self.forward_stem(x)

        out = []
        for pool_size, layer in zip((8, 4, 2, 1), (self.layer1, self.layer2, self.layer3, self.layer4)):
            x = layer(x)
            pool_x = F.avg_pool2d(x, (pool_size, pool_size), (pool_size, pool_size))
            out.append(bchw2bhwc(pool_x))
        return out

    def multi_level_forward_features(self, x):
        x = self.forward_stem(x)
        out = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            out.append(x)
        return out

    def forward_stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward_return_n_last_blocks(self, x, n, depths):
        output = []
        all_depths = sum(depths)
        block_idx = all_depths - n
        # stage 1 forward
        x = self.forward_stem(x)

        # layer1 to 4 decomposition
        layers = [getattr(self, f"layer{i}") for i in range(1, 5)]
        blocks = [b for l in layers for b in l]

        for idx, b in enumerate(blocks):
            x = b(x)
            if idx >= block_idx:
                output.append(self.avgpool(x).flatten(1))

        return torch.cat(output, dim=1)


def resnet18(*args, **kwargs):
    params = dict(
        block=BasicBlock,
        layers=[2, 2, 2, 2]
    )
    return ResnetWrapper(name="resnet18", **params)


def resnet34(*args, **kwargs):
    params = dict(
        block=BasicBlock,
        layers=[3, 4, 6, 3]
    )
    return ResnetWrapper(name="resnet34", **params)


def resnet50(*args, **kwargs):
    params = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3]
    )
    return ResnetWrapper(name="resnet50", **params)
