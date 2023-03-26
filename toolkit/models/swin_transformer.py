from typing import Any

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlock

__all__ = ['DINOHead', 'swin_tiny', 'swin_custom', 'LinearClassifier']


class SwinTransformerWrapper(SwinTransformer):
    def __init__(self, name, *args, **kwargs):
        super(SwinTransformerWrapper, self).__init__(*args, **kwargs)
        self.name = name
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        setattr(self, 'num_features', self.norm.normalized_shape[0])

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

    def forward_return_n_last_blocks(self, x, n=1, depths=[]):
        output = []
        all_depths = sum(depths)
        block_idx = all_depths - n
        for idx, f in enumerate(self.features):
            if isinstance(f, nn.Sequential) and isinstance(f[0], SwinTransformerBlock):
                for b in f:
                    x = b(x)
                    block_idx -= 1
                    if block_idx < 0:
                        output.append(self.avgpool(x.permute([0, 3, 1, 2])).squeeze())

            else:
                x = f(x)
        return torch.cat(output, dim=1)


def swin_tiny(*args, **kwargs):
    params = dict(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.0 if kwargs.get('is_teacher', False) else 0.2,
    )
    return SwinTransformerWrapper("swin_tiny", **params)


def swin_custom(*args, **kwargs):
    params = dict(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        window_size=[14, 14],
        stochastic_depth_prob=0.0 if kwargs.get('is_teacher', False) else 0.2,
    )
    return SwinTransformerWrapper("swin_custom", **params)
