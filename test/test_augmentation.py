import torch

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.mutli_transform import *
from toolkit.utils.plots import show, make_grid


class TestClass(object):
    def __init__(self):
        pass

    def test_mutil_crop_aug(self):
        transform = AlbumentationsLymphNodeV1(global_crops_scale=(0.75, 1.0),
                                              local_crops_scale=(0.25, 0.75),
                                              local_crops_number=8,
                                              local_crops_size=96,
                                              global_crops_size=224)
        dataset = KFoldLymphDataset(["../dataset_debug"], transform)
        for idx, d in enumerate(dataset):
            show(d['img'][:2], name=f"global_{idx}")
            show(make_grid(d['img'][2:], nrow=4), name=f"local_{idx}")
            if idx == 5:
                break

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith {m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
