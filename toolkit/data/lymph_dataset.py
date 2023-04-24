import glob
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torchvision

from tqdm import tqdm
from abc import ABC
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes
from sklearn.model_selection import StratifiedKFold
from toolkit.data.utils import IMG_FORMATS
from toolkit.utils.files import find_files
from toolkit.utils.python_utils import copy_attr
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


class LymphBaseDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 prefix="",
                 ):
        self.root = root
        self.class_to_idx = None
        self.classes = None
        self.labels = self.get_labels(img_path=self.root)

        self.prefix = prefix

    def get_labels(self, img_path):
        try:
            x = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                # Benign or Malignant
                classes, class_to_idx = self.find_classes(str(p))
                if self.class_to_idx is None:
                    self.classes = classes
                    self.class_to_idx = class_to_idx
                else:
                    if class_to_idx != self.class_to_idx and classes != self.classes:
                        raise ValueError(f"Two folder exists different class!.")

                for cls in classes:
                    # Find Patient id
                    patient_ids, _ = self.find_classes(str(p / cls))
                    for patient_id in patient_ids:
                        # Search Patient ID
                        search_p = p / cls / patient_id
                        if search_p.is_dir():
                            files = glob.glob(str(search_p / '**' / '*.*'), recursive=True)
                            for f in files:
                                d = dict(type_name=cls,
                                         label=class_to_idx[cls],
                                         patient_id=patient_id,
                                         im_file=f)
                                x.append(d)
                        else:
                            raise FileNotFoundError(f'{self.prefix}{p} does not exist')

            labels = [l for l in x if l['im_file'].split('.')[-1].lower() in IMG_FORMATS]
            return labels

        except Exception as e:
            raise FileNotFoundError(f'Error loading data from {img_path}\n') from e

    @staticmethod
    def find_classes(directory) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)


class KFoldLymphDataset(LymphBaseDataset):
    def __init__(self, root, transform=None, n_splits=3, shuffle=True, random_state=None):
        super().__init__(root)
        self.stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.transform = transform

    def generate_fold_dataset(self):
        labels = np.array([list(l.values()) for l in self.labels])
        for train_idx, test_idx in self.stratified_k_fold.split(
                labels[:, 0], labels[:, 1]):
            train_labels = list(np.array(self.labels)[train_idx])
            test_labels = list(np.array(self.labels)[test_idx])
            train_dataset, test_dataset = WrapperFoldDataset(train_labels), WrapperFoldDataset(test_labels)

            copy_attr(train_dataset, self, include=("class_to_idx", "classes"))
            copy_attr(test_dataset, self, include=("class_to_idx", "classes"))

            yield train_dataset, test_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item].copy()
        label['im_file'] = str(label['im_file'])
        label['img'] = pil_loader(label['im_file'])
        if self.transform:
            label['img'] = self.transform(label['img'])
        return label


class WrapperFoldDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        label = self.samples[item].copy()
        label['im_file'] = str(label['im_file'])
        label['img'] = pil_loader(label['im_file'])
        if self.transform:
            label['img'] = self.transform(label['img'])
        return label
