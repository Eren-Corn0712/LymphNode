from typing import Any

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlock

__all__ = ['swin_tiny', 'swin_custom']


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

        view_sizes = [inp.shape[-1] for inp in x]
        unique_sizes_count = torch.tensor(view_sizes).unique_consecutive(return_counts=True)[1]
        idx_crops = unique_sizes_count.cumsum(0)

        output = {}
        output_cls, output_fea, num_patch = [], [], []
        # Multi-view forward
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
        x = self.features(x)
        x_region = self.norm(x)
        x = self.permute(x_region)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x, x_region.flatten(1, 2)

    def forward_return_n_last_blocks(self, x, n=1, depths=[]):
        output = []
        all_depths = sum(depths)
        block_idx = all_depths - n

        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.Sequential):
                for l in layer:
                    x = l(x)
                    if isinstance(l, SwinTransformerBlock):
                        block_idx -= 1
                        if block_idx < 0:
                            output.append(self.avgpool(x.permute([0, 3, 1, 2])).squeeze())
            else:
                x = layer(x)

        return torch.cat(output, dim=1)


def swin_tiny(*args, **kwargs):
    params = dict(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        stochastic_depth_prob=0.2,
    )
    return SwinTransformerWrapper("swin_tiny", **params)


def swin_custom(*args, **kwargs):
    params = dict(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[1, 2, 4, 8],
        window_size=[14, 14],
        mlp_ratio=1.0,
        dropout=0.0,
        attention_dropout=0.0,
        stochastic_depth_prob=0.2,
    )
    return SwinTransformerWrapper("swin_custom", **params)
