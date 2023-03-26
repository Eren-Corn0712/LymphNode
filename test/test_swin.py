import torch
import torch.nn as nn
from toolkit.models.swin_transformer import swin_tiny, swin_custom
from toolkit.utils.torch_utils import select_device
from toolkit.utils.torch_utils import model_info


class TestSwinTransformer(object):
    def __init__(self):
        self.device = select_device('0')

    def test_swin_transformer(self):
        model = swin_custom().to(self.device)
        model_info(model, detailed=True, verbose=True, imgsz=224)

        model = swin_tiny().to(self.device)
        model_info(model, detailed=True, verbose=True, imgsz=224)

    def test_multi_input_forward(self):
        model = swin_custom().to(self.device)
        model.use_dense_prediction = True
        model.head = nn.Identity()
        model.head_dense = nn.Identity()
        b, c = 10, 3
        input = [torch.randn(b, c, i, i).to(self.device) for i in [224, 224, 96, 96, 96, 96]]
        output = model(input)
        assert output[0].shape[0] == b * (2 + 4), "Error"
        assert output[1].shape[0] == b * (49 * 2 + 9 * 4), "Local View Shape Error"

    def test_n_block_forward(self):
        model = swin_custom().to(self.device)
        depths = model.depths
        embed_dim = model.embed_dim
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d
        print(num_features)

        x = torch.randn(10, 3, 224, 224).to(self.device)
        for i in range(1,5):
            num_features_linear = sum(num_features[-i:])
            y = model.forward_return_n_last_blocks(x, n=i, depths=depths)
            print(f"y shape : {y.shape} num_features_linear = {num_features_linear}")
            assert y.shape == torch.Size([10, num_features_linear]), "Error"


    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


if __name__ == "__main__":
    TestSwinTransformer()()
