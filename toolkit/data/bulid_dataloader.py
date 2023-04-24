import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from toolkit.data.sampler import RASampler


def weight_sampler(dataset):
    targets = [d['label'] for d in dataset]
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def build_dataloader(args, backbone_dataset, train_dataset, test_dataset):
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(train_dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    backbone_data_loader = DataLoader(
        backbone_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=weight_sampler(train_dataset),
        batch_size=args.batch_size_per_gpu * 2,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size_per_gpu * 4,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return backbone_data_loader, train_loader, test_loader
