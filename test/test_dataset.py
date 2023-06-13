import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
from toolkit.data.lymph_dataset import LymphBaseDataset, KFoldLymphDataset

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import RandomSampler

DIR_DATASET = ["../dataset_clean"]


class TestClass(object):
    def __init__(self):
        pass

    def _test_lymph_dataset(self, *args, **kwargs):
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

    def test_dataloader(self):
        data_set = KFoldLymphDataset(DIR_DATASET)
        data_set.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        sampler = RandomSampler(data_source=data_set)

        data_loader = DataLoader(data_set, batch_size=32, sampler=sampler)
        indices = []
        for idx, batch in enumerate(data_loader):
            if idx <= 2:
                indices.extend(batch['idx'].tolist())
            if idx == 3:
                break

        print(indices)
        data_subset = Subset(data_set, indices)
        data_subset.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        data_sub_loader = DataLoader(data_subset, batch_size=32)
        for idx, batch in enumerate(data_sub_loader):
            print(batch['idx'])

    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
