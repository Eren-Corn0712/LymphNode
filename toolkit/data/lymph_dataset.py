import glob
import numpy as np
from typing import List, Tuple, Dict
from abc import ABC
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes
from sklearn.model_selection import StratifiedKFold
from toolkit.data.utils import IMG_FORMATS
from toolkit.utils import LOGGER
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

    def info(self):
        count_dict_label = {}
        for l in self.labels:
            count_dict_label.setdefault(l['type_name'], 0)
            count_dict_label[l['type_name']] += 1

        print(count_dict_label)

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
        X, X_dx = np.unique(labels[:, 2], return_index=True)  # ID
        y = labels[X_dx, 1]  # Label value
        for train_idx, test_idx in self.stratified_k_fold.split(X=X, y=y):
            train_key_label, train_key_id = y[train_idx], X[train_idx]
            test_key_label, test_key_id = y[test_idx], X[test_idx]

            train_labels = []
            for label, id in zip(train_key_label, train_key_id):
                train_labels += self.search_labels(int(label), id)

            test_labels = []
            for label, id in zip(test_key_label, test_key_id):
                test_labels += self.search_labels(int(label), id)

            self.check_dataset(train_labels, test_labels, 'patient_id')

            train_dataset, test_dataset = WrapperFoldDataset(train_labels), WrapperFoldDataset(test_labels)
            copy_attr(train_dataset, self, include=("class_to_idx", "classes"))
            copy_attr(test_dataset, self, include=("class_to_idx", "classes"))

            yield train_dataset, test_dataset

    def search_labels(self, label, id):
        x = []
        for l in self.labels:
            if l['label'] == label and l['patient_id'] == id:
                x.append(l)
        return x

    @staticmethod
    def check_dataset(train, test, key_name='patient_id'):
        train_check = set([t[key_name] for t in train])
        test_check = set([t[key_name] for t in test])
        merged_set = train_check.union(test_check)
        if len(merged_set) == len(train_check) + len(test_check):
            LOGGER.info("No duplicates found.")
        else:
            LOGGER.info("Duplicates found.")
            raise ValueError(f"Key {key_name} Duplicates found.")

        LOGGER.info(f"train data Patient id: {' '.join(s for s in train_check)}")
        LOGGER.info(f"test data Patient id: {' '.join(s for s in test_check)}")

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
