import os
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist

from toolkit.engine.base_train import BaseTrainer
from torch.utils.data import DataLoader, dataloader, distributed
from toolkit.data.lymph_dataset import KFoldLymphDataset

from toolkit.esvit import models
from toolkit.esvit.models import DINOHead
from toolkit.esvit.dino_loss import DDINOLoss
from toolkit.esvit.utils import (get_params_groups, cosine_scheduler, has_batchnorms, LARS, clip_gradients,
                                 cancel_gradients_last_layer, restart_from_checkpoint)
from toolkit.esvit.augment import DataAugmentationLymphNode, FineTuneAugmentation
from toolkit.esvit import models
from toolkit import colorstr, TQDM_BAR_FORMAT
from toolkit.utils import RANK
from toolkit.data.utils import PIN_MEMORY
from toolkit.utils.torch_utils import de_parallel, deepcopy
from tqdm import tqdm


class InfiniteDataLoader(dataloader.DataLoader):
    """Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EsVitTrainer(BaseTrainer):
    def __init__(self, cfg, overrides=None):
        super(EsVitTrainer, self).__init__(cfg=cfg, overrides=overrides)

    def get_dataset(self):
        self.trainset = KFoldLymphDataset(root=self.args.data_path, transform=None, n_splits=3, shuffle=True)

    def get_dataloader(self, *args, **kwargs):
        dataset = kwargs.get("dataset", None)
        mode = kwargs.get("mode", "train")
        rank = kwargs.get("rank", -1)

        shuffle = mode == 'train'
        batch = min(self.args.batch, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        workers = self.args.num_workers if mode == 'train' else self.args.num_workers * 2
        nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        return DataLoader(dataset=dataset,
                          batch_size=batch,
                          shuffle=shuffle and sampler is None,
                          num_workers=nw,
                          sampler=sampler,
                          pin_memory=PIN_MEMORY,
                          drop_last=True,
                          collate_fn=getattr(dataset, 'collate_fn', None),
                          worker_init_fn=seed_worker,
                          generator=generator), dataset

    def get_model(self, cfg=None, weights=None, verbose=True, **kwargs):
        if self.args.arch in models.__dict__.keys():
            model = models.__dict__[self.args.arch](is_teacher=kwargs.get("is_teacher", False))
        else:
            raise ValueError(f"Unknown Arch {self.args.arch}.")

        setattr(model, "use_dense_prediction", self.args.use_dense_prediction)
        setattr(model, "head", DINOHead(model.norm.normalized_shape[0],
                                        self.args.out_dim,
                                        self.args.use_bn_in_head,
                                        self.args.norm_last_layer))

        if self.args.use_dense_prediction:
            setattr(model, "head_dense", DINOHead(model.norm.normalized_shape[0],
                                                  self.args.out_dim,
                                                  self.args.use_bn_in_head))
        return model

    def _setup_train(self, rank, world_size):
        self.teacher = self.get_model(is_teacher=True)
        self.student = self.get_model(is_teacher=False)
        pass

    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)
        self._setup_train(rank, world_size)
