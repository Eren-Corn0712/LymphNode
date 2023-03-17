import torch
import torchvision.transforms as transforms
from toolkit.data.lymph_dataset import LymphBaseDataset, KFoldLymphDataset
from pathlib import Path

DIR_DATASET = "../dataset"


class TestClass(object):
    def __init__(self):
        pass

    def test_lymph_dataset(self, *args, **kwargs):
        LymphBaseDataset(DIR_DATASET)

    def test_k_fold_lymph_dataset(self, *args, **kwargs):
        dataset = KFoldLymphDataset(DIR_DATASET)
        for train_dataset, test_dataset in dataset.generate_data_splits():
            train_dataset.transform = transforms.Compose([transforms.ToTensor()])
            for i in train_dataset:
                print(i)

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
