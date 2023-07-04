from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import MLP

from toolkit.models.nn.block import Bottleneck
from toolkit.models.nn.conv import *
from toolkit.models.nn.transformer import LayerNorm2d
from toolkit.utils import LOGGER


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Interpolate(nn.Module):
    def __init__(self,
                 size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None,
                 antialias=False
                 ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners,
                             self.recompute_scale_factor,
                             self.antialias)


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

    def forward(self, output_cls: List, output_fea: List, num_patch: List):
        batch_size = output_cls[0].shape[0] // 2
        output = []
        for cls, fea, patch in zip(output_cls, output_fea, num_patch):
            # cls size: 2B, K
            # fea size: (2BN + 8Bn), K
            # 2BN = (BN,) * 2
            # 8BN = (BN,) * 8
            n_sample = cls.shape[0] // batch_size  # 2 or 8
            split_size = patch * batch_size  # BN

            cls = self.global_mlp(cls)
            fea = self.local_mlp(fea)

            fea = torch.split(fea, [split_size] * n_sample, dim=0)  #
            fea = torch.cat([f.reshape(batch_size, patch, -1) for f in fea], dim=0)

            x = self.next_to_last(cls[:, None, :] + fea)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
            batch_size, num_fea, channel = x.shape
            output.append(x.reshape(batch_size * num_fea, channel))

        return torch.cat(output)


class TransformerHead(nn.Module):
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
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=bottleneck_dim,
                                                       batch_first=True)
            layers.append(nn.TransformerEncoder(encoder_layer, num_layers=num_layers - 2))
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        output = []
        for i in x:
            i = self.mlp(i)
            i = nn.functional.normalize(i, dim=-1, p=2)
            i = self.last_layer(i)
            output.append(i)
        return output


class GELU(nn.Module):
    def __init__(self, inplace=False, approximate="none"):
        super().__init__()
        self.inplace = inplace
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(repr(self.approximate))


class MultiLevelHead(nn.Module):
    def __init__(
            self,
            in_dim: List,
            out_dim,
            use_bn=False,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=2048,
            bottleneck_dim=256

    ):
        super().__init__()
        self.nl = len(in_dim)  # number of projection layers
        self.mlp = nn.ModuleList(
            (MLP(
                in_channels=in_d,
                hidden_channels=[hidden_dim] * (num_layers - 2) + [bottleneck_dim],
                norm_layer=nn.BatchNorm1d if use_bn else None,
                activation_layer=GELU,
            ) for in_d in in_dim
            )
        )

        self.apply(init_weights)
        self.last_layers = nn.ModuleList(
            nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)) for _ in range(self.nl))
        for layer in self.last_layers:
            layer.weight_g.data.fill_(1)

        if norm_last_layer:
            for layer in self.last_layers:
                layer.weight_g.requires_grad = False

    def forward(self, x):
        output = [[] for _ in range(len(x))]
        for i in range(self.nl):
            output[i] = self.mlp[i](x[i])
            output[i] = nn.functional.normalize(output[i], dim=-1, p=2)
            output[i] = self.last_layers[i](output[i])
        return output


class CrossLevelHead(nn.Module):
    def __init__(
            self,
            in_dim: List,
            scale: float,
            learnable_sample=True,
    ):
        super().__init__()

        LOGGER.info(f"Head: {self.__class__.__name__}")
        LOGGER.info(f"in_dim: {in_dim}")
        LOGGER.info(f"scale: {scale}")
        LOGGER.info(f"learnable_sample: {learnable_sample}")

        self.fusion_layer = nn.ModuleList(
            SelfRelationModule(
                c1=in_dim[i],
                c2=in_dim[i + 1],
                scale=scale,
                learnable_sample=learnable_sample
            )
            for i in range(len(in_dim) - 1)
        )


    def forward(self, x):
        output = []
        for i, layer in enumerate(self.fusion_layer):
            layer_out = layer(x[i], x[i + 1])
            output.append(layer_out)

        return output


class SelfRelationModule(nn.Module):
    def __init__(
            self,
            c1: int,
            c2: int,
            scale: float,
            learnable_sample: True):
        super().__init__()
        self.c1_scale = int(c1 * scale)
        self.c2_scale = int(c2 * scale)

        self.norm1 = LayerNorm2d(self.c1_scale)
        self.norm2 = LayerNorm2d(self.c2_scale)

        self.norm3 = LayerNorm2d(self.c1_scale)
        self.norm4 = LayerNorm2d(self.c2_scale)

        self.proj1 = Bottleneck(c1, self.c1_scale, k=(1, 1))
        self.proj2 = Bottleneck(c2, self.c2_scale, k=(1, 1))

        if learnable_sample is True:
            self.up_sa = ConvTranspose(c2, c2, 4, 2, 1, act=nn.GELU())
            self.down_sa = Conv(c1, c1, 3, 2, act=nn.GELU())
        else:
            self.up_sa = Interpolate(scale_factor=2)
            self.down_sa = Interpolate(scale_factor=0.5)

        self.apply(init_weights)

    def forward(self, x, y):
        down_x, up_y = self.down_sa(x), self.up_sa(y)
        x, y, down_x, up_y = self.proj1(x), self.proj2(y), self.proj1(down_x), self.proj2(up_y)

        # normalizer layer
        x = self.norm1(x)
        y = self.norm2(y)
        down_x = self.norm3(down_x)
        up_y = self.norm4(up_y)

        # Some Variables
        (b, c1, h1, w1), (b, c2, h2, w2) = x.shape, y.shape
        dim1, dim2 = h1 * w1, h2 * w2
        ch_scale1, ch_scale2 = (dim1 ** -0.5), (dim2 ** -0.5)

        # Reshape
        x, y = x.view(b, c1, dim1), y.view(b, c2, dim2)
        down_x, up_y = down_x.view(b, c1, dim2), up_y.view(b, c2, dim1)

        # Attention
        # Pre scale
        x = x * ch_scale1
        y = y * ch_scale2

        dot1 = torch.bmm(x, up_y.transpose(-1, -2))
        dot2 = torch.bmm(y, down_x.transpose(-1, -2))

        attn1 = F.softmax(dot1, dim=-1)
        attn2 = F.softmax(dot2, dim=-1)

        out1 = torch.bmm(attn1, y)
        out2 = torch.bmm(attn2, x)

        return out1, out2


class SelfRelationHeadV2(nn.Module):
    def __init__(
            self,
            in_dim,
            scale=1.0,
            use_bn=False,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=1024,
            bottleneck_dim=128,

    ):
        super().__init__()
        self.fusion_layer = nn.ModuleList(
            SelfRelationModuleV2(
                c1=in_dim[i],
                c2=in_dim[i + 1],
                scale=scale,
                use_bn=use_bn,
                norm_last_layer=norm_last_layer,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim)
            for i in range(len(in_dim) - 1)
        )

    def forward(self, x):
        output = []
        for i, layer in enumerate(self.fusion_layer):
            layer_out = layer(x[i], x[i + 1])
            output.append(layer_out)

        return output


class SelfRelationModuleV2(SelfRelationModule):
    def __init__(
            self,
            c1: int,
            c2: int,
            scale: float,
            use_bn=False,
            norm_last_layer=True,
            num_layers=3,
            hidden_dim=1024,
            bottleneck_dim=128
    ):
        super().__init__(c1, c2, scale)

        self.norm1 = LayerNorm2d(c1)
        self.norm2 = LayerNorm2d(c2)

        self.norm3 = LayerNorm2d(c1)
        self.norm4 = LayerNorm2d(c2)

        self.proj1 = MLP(
            in_channels=c1,
            hidden_channels=[hidden_dim] * (num_layers - 2) + [bottleneck_dim],
            norm_layer=nn.BatchNorm1d if use_bn else None,
            activation_layer=GELU)

        self.proj2 = MLP(
            in_channels=c2,
            hidden_channels=[hidden_dim] * (num_layers - 2) + [bottleneck_dim],
            norm_layer=nn.BatchNorm1d if use_bn else None,
            activation_layer=GELU)

        self.apply(init_weights)
        self.last_layer1 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, self.c1_scale, bias=False))
        self.last_layer1.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer1.weight_g.requires_grad = False

        self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, self.c2_scale, bias=False))
        self.last_layer2.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer2.weight_g.requires_grad = False

    def forward(self, x, y):
        # up-sample and down-sample
        down_x, up_y = self.down_sa(x), self.up_sa(y)

        # normalizer layer
        x = self.norm1(x)
        y = self.norm2(y)
        down_x = self.norm3(down_x)
        up_y = self.norm4(up_y)

        # Some Variables
        (b, c1, h1, w1), (b, c2, h2, w2) = x.shape, y.shape
        dim1, dim2 = h1 * w1, h2 * w2
        ch_scale1, ch_scale2 = (dim1 ** -0.5), (dim2 ** -0.5)

        # Reshape
        x, y = x.view(b, c1, dim1), y.view(b, c2, dim2)
        down_x, up_y = down_x.view(b, c1, dim2), up_y.view(b, c2, dim1)

        # Attention
        # Pre scale
        x = x * ch_scale1
        y = y * ch_scale2

        out1 = self.attn_opt(x, up_y, y)
        out2 = self.attn_opt(y, down_x, x)

        out1 = self.proj1(out1.view(b, c1, dim2).permute(0, 2, 1).contiguous().view(b * dim2, c1))  # b c hw -> b hw c
        out2 = self.proj2(out2.view(b, c2, dim1).permute(0, 2, 1).contiguous().view(b * dim1, c2))  # b c hw -> b hw c

        out1 = nn.functional.normalize(out1, dim=-1, p=2)
        out1 = self.last_layer1(out1)

        out2 = nn.functional.normalize(out2, dim=-1, p=2)
        out2 = self.last_layer2(out2)

        return out1.view(b, dim2, self.c1_scale), out2.view(b, dim1, self.c2_scale)

    @staticmethod
    def attn_opt(query, key, value):
        dot = torch.bmm(query, key.transpose(-1, -2))
        attn = F.softmax(dot, dim=-1)
        out = torch.bmm(attn, value)
        return out


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
