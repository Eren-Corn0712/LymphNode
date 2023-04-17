import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Permute
from torchvision.models.vision_transformer import Encoder


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


class AttnHead(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            use_bn=False,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=2048,
            bottleneck_dim=256,
            num_heads=4,
    ):
        super().__init__()

        self.encoder_layer = Encoder(seq_length=(224 // 32) ** 2,
                                     num_layers=num_layers,
                                     num_heads=num_heads,
                                     hidden_dim=in_dim,
                                     mlp_dim=out_dim,
                                     dropout=0.1,
                                     attention_dropout=0.1)

    def forward(self, x):
        # Torchvision supoort input size is b,s,c
        x = self.encoder_layer(x)
        return x
