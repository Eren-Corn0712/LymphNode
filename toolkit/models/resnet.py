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
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        if self.use_dense_prediction:
            start_idx = 0

            for end_idx in idx_crops:
                _out_cls, _out_fea = self.forward_features(torch.cat(x[start_idx: end_idx]))
                B, N, C = _out_fea.shape

                if start_idx == 0:
                    output_cls = _out_cls
                    output_fea = _out_fea.reshape(B * N, C)
                    npatch = [N]
                else:
                    output_cls = torch.cat((output_cls, _out_cls))
                    output_fea = torch.cat((output_fea, _out_fea.reshape(B * N, C)))
                    npatch.append(N)
                start_idx = end_idx

            return self.head(output_cls), self.head_dense(output_fea), output_fea, npatch

        else:
            start_idx = 0
            for end_idx in idx_crops:
                _out = super().forward(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            # Run the head forward on the concatenated features.
            return self.head(output)

    def forward_features(self, x):

        x_region = self.forward_feature_map(x)
        H, W = x_region.shape[-2], x_region.shape[-1]

        x = self.avgpool(x_region)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, rearrange(x_region, 'b c h w -> b (h w) c', h=H, w=W)

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

    def forward_return_n_last_blocks(self, x, n=1, depths=[]):
        output = []
        all_depths = sum(depths)
        block_idx = all_depths - n
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for i in range(1, 5):
            f = getattr(self, f"layer{i}")
            for b in f:
                x = b(x)
                block_idx -= 1
                if block_idx < 0:
                    output.append(self.avgpool(x).squeeze())

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
