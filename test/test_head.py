import unittest

import torch
from toolkit.utils.torch_utils import select_device
from toolkit.models.head import DINOHead,MixDINOHead,TransformerHead


class TestSwinTransformer(unittest.TestCase):

    def setUp(self) -> None:
        self.b = 10
        self.c = 3
        self.device = select_device("cuda:0")
        self.input = torch.randn(self.b, 49 * 49, 768, device=self.device)

    def test_head_forward(self):
        head = AttnHead(768, 2048).to(self.device)
        output = head(self.input)

        self.assertEqual(first=output.shape,
                         second=torch.Size([self.b, 2048]),
                         msg="Successful pass!")


if __name__ == "__main__":
    unittest.main()
