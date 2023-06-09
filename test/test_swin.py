import unittest

import torch
import torch.nn as nn
from toolkit.models.swin_transformer import swin_tiny, swin_custom
from toolkit.utils.torch_utils import select_device
from toolkit.utils.torch_utils import model_info


class TestSwinTransformer(unittest.TestCase):

    def setUp(self) -> None:
        self.b = 10
        self.c = 3
        self.device = select_device("cuda:0")
        self.input = [torch.randn(self.b, self.c, i, i).to(self.device) for i in [224] * 2 + [96] * 8]
        self.global_h = int(224 / 32)
        self.global_w = int(224 / 32)

        self.local_h = int(96 / 32)
        self.local_w = int(96 / 32)

    def test_multi_forward_pass(self):
        model = swin_custom().to(self.device)
        model.head = nn.Identity()
        model.dense_head = nn.Identity()
        model.use_dense_prediction = True
        output = model(self.input)
        self.assertEqual(output[0].shape, torch.Size((len(self.input) * self.b, 768)))
        self.assertEqual(output[1].shape, torch.Size(
            (len(self.input[:2]) * self.b * self.global_h * self.global_w + len(
                self.input[2:]) * self.b * self.local_h * self.local_w, 768)))

    def test_n_block_forward(self):
        model = swin_custom().to(self.device)
        depths = model.depths
        embed_dim = model.embed_dim
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        x = torch.randn(10, 3, 224, 224).to(self.device)
        for i in range(1, 5):
            num_features_linear = sum(num_features[-i:])
            y = model.forward_return_n_last_blocks(x, n=i, depths=depths)
            assert y.shape == torch.Size([10, num_features_linear]), "Error"


if __name__ == "__main__":
    unittest.main()
