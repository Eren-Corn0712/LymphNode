import torch
import torch.nn as nn
from toolkit.models.resnet import resnet18, resnet34, resnet50
from toolkit.utils.torch_utils import select_device
from toolkit.utils.torch_utils import model_info


class TestResNet(object):
    def __init__(self):
        self.device = select_device('0')

    def test_resnet18(self):
        model = resnet18().to(self.device)
        model_info(model, detailed=True, verbose=True, imgsz=224)

    def test_multi_level_forward(self):
        model = resnet18().to(self.device)
        x = torch.randn(10, 3, 224, 224).to(self.device)
        out = model.multi_level_forward_features(x)
        print(
            "ok"
        )

    def test_multi_level_forward_in_multi_view(self):
        b, c = 10, 3
        input = [torch.randn(b, c, i, i).to(self.device) for i in [224, 224, 96, 96, 96, 96]]

        model = resnet18().to(self.device)
        model.multi_level = True
        output = model(input)

    def _test_multi_input_forward(self):
        model = resnet18().to(self.device)
        model.use_dense_prediction = True
        model.head = nn.Identity()
        model.dense_head = nn.Identity()
        b, c = 10, 3
        input = [torch.randn(b, c, i, i).to(self.device) for i in [224, 224, 96, 96, 96, 96]]
        l = len(input)
        global_fea = int(224 / 32)
        local_fea = int(96 / 32)
        output = model(input)
        assert b * l == output[0].shape[0]
        assert b * 2 * (global_fea ** 2) + b * (l - 2) * (local_fea ** 2) == output[1].shape[0]

    def _test_n_block_forward(self):
        model = resnet50().to(self.device)
        depths = model.layers
        embed_dim = model.conv1.out_channels * model.layer1[-1].expansion
        num_features = []

        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d
        print(num_features)

        x = torch.randn(10, 3, 224, 224).to(self.device)
        for i in range(1, 5):
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
    TestResNet()()
