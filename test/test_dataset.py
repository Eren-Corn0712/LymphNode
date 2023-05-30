import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from toolkit.data.lymph_dataset import LymphBaseDataset, KFoldLymphDataset
from pathlib import Path
from tqdm import tqdm

DIR_DATASET = ["../dataset"]


class TestClass(object):
    def __init__(self):
        pass

    def test_lymph_dataset(self, *args, **kwargs):
        data_set = LymphBaseDataset(DIR_DATASET)
        print(data_set.info())
        print(len(data_set))
    def _test_k_fold_lymph_dataset(self, *args, **kwargs):
        dataset1 = KFoldLymphDataset(DIR_DATASET, random_state=0)
        dataset2 = KFoldLymphDataset(DIR_DATASET, random_state=0)
        z = zip(dataset1.generate_fold_dataset(), dataset2.generate_fold_dataset())
        for a, b in z:
            print(a[0] == b[0])
            print(a[1] == b[1])

    def _test_compute_mean_var(self):
        dataset = KFoldLymphDataset(DIR_DATASET)
        dataset.transform = transforms.Compose([transforms.ToTensor()])
        print(f"len = {len(dataset)}")
        data_list = []
        for data in tqdm(dataset):
            data_list.append(data['img'].flatten())

        data_list = torch.cat(data_list)
        mean = torch.mean(data_list)
        std = torch.std(data_list)
        print("Mean:", mean)
        print("std:", std)

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
