import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Permute


class LinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.linear(x)))


class CAREHead(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_heads=4,
            dim_feedforward=2048,
            hidden_dim=2048,
            bottleneck_dim=256,
            norm_last_layer=True,
    ):
        super().__init__()
        layers = []
        layers += [nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation=F.gelu)] * 2

        layers.append(Permute([0, 2, 1]))  # b, h * w, c -> b, c, h *  w
        layers.append(nn.AdaptiveAvgPool1d(1))  # b, c, h * w -> b, c, 1
        layers.append(nn.Flatten(1))  # b, c, 1 -> b, c

        layers.append(LinearLayer(in_dim, hidden_dim))
        layers.append(LinearLayer(hidden_dim, hidden_dim))
        layers.append(LinearLayer(hidden_dim, bottleneck_dim))
        self.layers = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.layers(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x
