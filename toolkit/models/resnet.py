import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from einops import rearrange, repeat

__all__ = ["resnet18", "resnet34", "resnet50"]


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

        batch_size = x[0].shape[0]
        view_sizes = [inp.shape[-1] for inp in x]
        unique_sizes_count = torch.tensor(view_sizes).unique_consecutive(return_counts=True)[1]
        idx_crops = unique_sizes_count.cumsum(0)

        output = {}
        output_cls, output_fea, num_patch = [], [], []
        # Multi view forward
        for start_idx, end_idx in zip([0] + idx_crops[:-1].tolist(), idx_crops.tolist()):
            out_cls, out_fea = self.forward_features(torch.cat(x[start_idx: end_idx]))

            # Concatenate the features across patches
            batch_size, num_fea, channel = out_fea.shape
            output_cls.append(out_cls)
            output_fea.append(out_fea.reshape(batch_size * num_fea, channel))

            # Record batch size
            num_patch.append(num_fea)
            del out_cls, out_fea

        if hasattr(self, "head") and output_cls:
            output["head"] = getattr(self, 'head')(torch.cat(output_cls))

        if hasattr(self, "dense_head") and output_fea:
            output["dense_head"] = getattr(self, 'dense_head')(torch.cat(output_fea))

        if hasattr(self, "mix_head") and output_cls and output_fea:
            output["mix_head"] = getattr(self, 'mix_head')(
                output_cls, output_fea, num_patch)

        output["output_fea"] = torch.cat(output_fea)
        output["num_patch"] = num_patch
        return output

    def forward_features(self, x):
        x_region = self.forward_feature_map(x)

        h, w = x_region.shape[-2:]
        x = self.avgpool(x_region)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, rearrange(x_region, 'b c h w -> b (h w) c', h=h, w=w)

    def forward_feature_map(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_return_n_last_blocks(self, x, n, depths):
        output = []
        all_depths = sum(depths)
        block_idx = all_depths - n
        # stage 1 forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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
