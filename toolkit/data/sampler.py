import math

import torch
import torch.utils.data
import torch.distributed as dist
from torch.utils.data import (RandomSampler, WeightedRandomSampler, SequentialSampler, Sampler)
from torch.utils.data.distributed import (DistributedSampler)
from typing import Dict, Optional
from toolkit.utils import LOGGER


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class WeightedRandomDistributedSampler(DistributedSampler):
    def __init__(self, weights, dataset, replacement=True,
                 generator=None,
                 # DistributedSampler __init__
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))

        self.weights = weights_tensor
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # [1,3,5,7,9] or [0,2,4,6,8,10] if rank=2
        assert len(indices) == self.num_samples
        weights_indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=self.generator)
        sampled_indices = weights_indices[indices]
        return iter(sampled_indices.tolist())


def compute_sample_weights(dataset):
    class_to_idx: Dict = dataset.class_to_idx
    class_sample_count = {k: 0 for k in class_to_idx.keys()}
    for i in dataset:
        class_sample_count[i["type_name"]] += 1

    class_sample_weight = {k: 1 / v for k, v in class_sample_count.items()}
    weights = torch.tensor([
        class_sample_weight[i["type_name"]] for i in dataset
    ])
    assert len(weights) == len(dataset), "The sample weights number is not equal the data!"
    return weights


def creat_sampler(args, train_dataset, test_dataset):
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps)
        elif hasattr(args, "weighted_sampler") and args.weighted_sampler:
            weights = compute_sample_weights(train_dataset)
            train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset))
        else:
            train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        if hasattr(args, "weighted_sampler") and args.weighted_sampler:
            weights = compute_sample_weights(train_dataset)
            train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset))
        else:
            train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)

    LOGGER.info(f"Training Sampler : {train_sampler.__class__.__name__}")
    LOGGER.info(f"Test Sampler : {test_sampler.__class__.__name__}")
    return train_sampler, test_sampler
