from typing import List

import torch

import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DINOHead(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            use_bn=False,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=2048,
            bottleneck_dim=256
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MixDINOHead(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            use_bn=False,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=2048,
            bottleneck_dim=256
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.global_mlp = nn.Linear(in_dim, bottleneck_dim)
            self.local_mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            global_layers = [nn.Linear(in_dim, hidden_dim)]
            local_layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                global_layers.append(nn.BatchNorm1d(hidden_dim))
                local_layers.append(nn.BatchNorm1d(hidden_dim))
            global_layers.append(nn.GELU())
            local_layers.append(nn.GELU())

            global_layers.append(nn.Linear(hidden_dim, hidden_dim))
            local_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                global_layers.append(nn.BatchNorm1d(hidden_dim))
                local_layers.append(nn.BatchNorm1d(hidden_dim))
            global_layers.append(nn.GELU())
            local_layers.append(nn.GELU())

            self.global_mlp = nn.Sequential(*global_layers)
            self.local_mlp = nn.Sequential(*local_layers)

        self.next_to_last = nn.Linear(hidden_dim, bottleneck_dim)
        self.apply(init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, output_cls: List, output_fea: List, num_patch: List):
        batch_size = output_cls[0].shape[0] // 2
        output = []
        for cls, fea, patch in zip(output_cls, output_fea, num_patch):
            # cls: 2B, K
            # fea: (2BN + 8Bn), K
            n_sample = cls.shape[0] // batch_size  # 2 or 8
            split_size = patch * batch_size  # BN
            fea = torch.split(fea, [split_size] * n_sample, dim=0)
            fea = torch.cat([f.reshape(batch_size, patch, -1) for f in fea], dim=0)

            cls = self.global_mlp(cls)
            fea = self.local_mlp(fea)

            x = self.next_to_last(cls[:, None, :] * fea)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
            batch_size, num_fea, channel = x.shape
            output.append(x.reshape(batch_size * num_fea, channel))

        return torch.cat(output)


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
