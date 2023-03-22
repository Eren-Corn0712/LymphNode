import torch
import torch.nn as nn
from toolkit.esvit.models import swin_s
from toolkit.utils.torch_utils import select_device


class TestSwinTransformer(object):
    def __init__(self):
        self.device = select_device('0')

    def test_multi_input_forward(self):
        model = swin_s().to(self.device)
        model.use_dense_prediction = True
        model.head = nn.Identity()
        model.head_dense = nn.Identity()
        b, c = 10, 3
        input = [torch.randn(b, c, i, i).to(self.device) for i in [224, 224, 96, 96, 96, 96]]
        output = model(input)
        assert output[0].shape[0] == b * (2 + 4), "Error"
        assert output[1].shape[0] == b * (49 * 2 + 9 * 4), "Local View Shape Error"

    def test_n_block_forward(self):
        model = swin_s().to(self.device)
        base_c = 96
        x = torch.randn(10, 3, 224, 224).to(self.device)
        y = model.forward_return_n_last_block(x, n=4, return_patch_avgpool=False, depth=[2, 2, 18, 2])
        assert y.shape == torch.Size([10, (8 + 4 + 2 + 1) * base_c, 1, 1]), "Error"
        y = model.forward_return_n_last_block(x, n=3, return_patch_avgpool=False, depth=[2, 2, 18, 2])
        assert y.shape == torch.Size([10, (8 + 4 + 2) * base_c, 1, 1]), "Error"
        y = model.forward_return_n_last_block(x, n=2, return_patch_avgpool=False, depth=[2, 2, 18, 2])
        assert y.shape == torch.Size([10, (8 + 4) * base_c, 1, 1]), "Error"
        y = model.forward_return_n_last_block(x, n=1, return_patch_avgpool=False, depth=[2, 2, 18, 2])
        assert y.shape == torch.Size([10, 8 * base_c, 1, 1]), "Error"

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


if __name__ == "__main__":
    TestSwinTransformer()()
