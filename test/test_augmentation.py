import torch

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.mutli_transform import DataAugmentationLymphNode, AlbumentationsLymphNode
from toolkit.utils.plots import show


class TestClass(object):
    def __init__(self):
        pass

    def test_data_augmentation_lymph_node(self):
        transform = DataAugmentationLymphNode(global_crops_scale=(0.95, 1.0),
                                              local_crops_scale=(0.50, 0.75),
                                              local_crops_number=8,
                                              global_crops_size=224,
                                              local_crops_size=96)
        dataset = KFoldLymphDataset(root=["../dataset"], transform=transform)
        for idx, d in enumerate(dataset):
            show(d['img'][:2])
            show(d['img'][2:])
            if idx == 5:
                break

    def _test_albumentations(self):
        transform = AlbumentationsLymphNode(global_crops_scale=(0.95, 1.0),
                                            local_crops_scale=(0.50, 0.75),
                                            local_crops_number=8,
                                            global_crops_size=224,
                                            local_crops_size=96)
        dataset = KFoldLymphDataset(root=["../dataset"], transform=transform)
        for idx, d in enumerate(dataset):
            show(d['img'][:2])
            show(d['img'][2:])
            if idx == 10:
                break

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
