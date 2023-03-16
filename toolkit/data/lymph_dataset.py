import glob
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torchvision

from tqdm import tqdm
from abc import ABC
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset, find_classes
from sklearn.model_selection import StratifiedKFold
from toolkit.data.utils import IMG_FORMATS, find_files


class LymphBaseDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 prefix=''
                 ):
        self.root = root
        self.classes, self.class_to_idx = self.find_classes(self.root)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.patient_id, self.patient_id_to_class = self.find_patient_id()
        self.prefix = prefix

    def find_patient_id(self):
        patient_id, patient_id_to_class = [], {}
        for k, v in self.class_to_idx.items():
            sub_dir = Path(self.root) / k

            p_id, p_id_to_idx = self.find_classes(str(sub_dir))

            patient_id += p_id
            patient_id_to_class.update({c: v for c, _ in p_id_to_idx.items()})

        return patient_id, patient_id_to_class

    @staticmethod
    def find_classes(directory) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)


class KFoldLymphDataset(LymphBaseDataset):
    def __init__(self, root, n_splits=5, shuffle=False, random_state=None):
        super().__init__(root)
        self.stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def generate_data_splits(self):
        patient_id = list(self.patient_id_to_class.keys())
        classes = list(self.patient_id_to_class.values())
        for train_idx, test_idx in self.stratified_k_fold.split(patient_id, classes):
            train_id, train_labels = np.array(patient_id)[train_idx], np.array(classes)[train_idx]
            test_id, test_labels = np.array(patient_id)[test_idx], np.array(classes)[test_idx]
            train_f = self.get_sample(train_id, train_labels)
            test_f = self.get_sample(test_id, test_labels)
            yield WrapperKFoldDataset(train_f), WrapperKFoldDataset(test_f)

    def get_sample(self, ids, labels):
        f = []
        for id, label in zip(ids, labels):
            p = Path(self.root) / self.idx_to_class[label] / id
            im_files = find_files(str(p), 'jpg', recursive=True)
            for im_file in im_files:
                f += [im_file, label]
        return f


class WrapperKFoldDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        im_file, label = self.samples[item].copy()
        label = {}
        if self.transform:
            pass

        return label


if __name__ == "__main__":
    lymph_dataset = LymphBaseDataset("/home/corn/PycharmProjects/LymphNode/lymph-node-1.20-square")
