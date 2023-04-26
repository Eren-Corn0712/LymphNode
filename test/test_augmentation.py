import torch

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.augmentations import DataAugmentationLymphNodeOverlapping, DataAugmentationLymphNode
from toolkit.utils.plots import show

class TestClass(object):
    def __init__(self):
        pass

    def test_mutil_crop_aug(self):
        transform = DataAugmentationLymphNodeOverlapping(global_crops_scale=(0.4, 1.0),
                                                         local_crops_scale=(0.95, 1.0),
                                                         local_crops_number=[8, ],
                                                         local_crops_size=[96, ])
        dataset = KFoldLymphDataset(["../dataset"], transform)
        for idx, d in enumerate(dataset):
            show(d['img'][:2])
            show(d['img'][2:])
            if idx == 0:
                break


    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
