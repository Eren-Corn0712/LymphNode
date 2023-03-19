from typing import Any

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer

__all__ = ['DINOHead', 'swin_s']


class SwinTransformerWrapper(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super(SwinTransformerWrapper, self).__init__(*args, **kwargs)
        self.use_dense_prediction = kwargs.get("use_dense_prediction", False)

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        if self.use_dense_prediction:
            output_cls, output_fea, n_patch = None, None, []
            start_idx = 0

            for end_idx in idx_crops:
                _out_cls, _out_fea = self.forward_features(torch.cat(x[start_idx: end_idx]))
                B, N, C = _out_fea.shape

                if start_idx == 0:
                    output_cls = _out_cls
                    output_fea = _out_fea.reshape(B * N, C)
                    n_patch = [N]
                else:
                    output_cls = torch.cat((output_cls, _out_cls))
                    output_fea = torch.cat((output_fea, _out_fea.reshape(B * N, C)))
                    n_patch.append(N)
                start_idx = end_idx

            return self.head(output_cls), self.head_dense(output_fea), output_fea, n_patch
        else:
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.forward_features(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            # Run the head forward on the concatenated features.
            return self.head(output)

    def forward_features(self, x):
        x = self.features(x)
        x_region = self.norm(x)
        x = self.permute(x_region)
        x = self.avgpool(x)
        x = self.flatten(x)

        if self.use_dense_prediction:
            return x, x_region.flatten(1, 2)
        else:
            return x


def swin_s(*args, **kwargs):
    params = {
        'patch_size': [4, 4],
        'embed_dim': 96,
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': [7, 7],
        'stochastic_depth_prob': 0.0 if kwargs.get('is_teacher') else 0.3,
    }
    return SwinTransformerWrapper(**params)


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
