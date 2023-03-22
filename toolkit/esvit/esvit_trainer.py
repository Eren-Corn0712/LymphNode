import os
import time

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
import random
from toolkit.engine.base_train import BaseTrainer
from torch.utils.data import DataLoader, dataloader, distributed

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.esvit.models import DINOHead
from toolkit.esvit.dino_loss import DDINOLoss, DINOLoss
from toolkit.esvit.utils import (get_params_groups, cosine_scheduler, has_batchnorms, LARS, clip_gradients,
                                 cancel_gradients_last_layer, restart_from_checkpoint)
from toolkit.esvit.augment import get_transform
from toolkit.esvit import models
from toolkit import colorstr, TQDM_BAR_FORMAT
from toolkit.utils import RANK
from toolkit.data.utils import PIN_MEMORY
from toolkit.utils.torch_utils import de_parallel, deepcopy, model_info
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

        self.loss_names = ["view", "local"]

    def get_dataset(self):
        self.trainset = KFoldLymphDataset(root=self.args.data_path, transform=None, n_splits=3, shuffle=True)

    def get_transform(self):
        return get_transform(self.args)

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

    def get_loss(self):
        if self.args.use_dense_prediction:
            # Both view and region level tasks are considered
            self.dino_loss = DDINOLoss(
                self.args.out_dim,
                sum(self.args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
                self.args.warmup_teacher_temp,
                self.args.teacher_temp,
                self.args.warmup_teacher_temp_epochs,
                self.args.epochs,
            ).to(self.device)
        else:
            # Only view level task is considered
            self.dino_loss = DINOLoss(
                self.args.out_dim,
                sum(self.args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
                self.args.warmup_teacher_temp,
                self.args.teacher_temp,
                self.args.warmup_teacher_temp_epochs,
                self.args.epochs,
            ).to(self.device)

    def build_optimizer(self):
        params_groups = get_params_groups(self.student)
        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif self.args.optimizer == "lars":
            self.optimizer = LARS(params_groups)  # to use with convnet and large batches
        else:
            raise ValueError(f"{self.args.optimizer} is not Supported ")

        s = f"regularized: {len(params_groups[0]['params'])} not_regularized: {len(params_groups[1]['params'])}"

        self.log(
            f"{colorstr('optimizer:')} {type(self.optimizer).__name__} {s} with parameter groups ")

    def _setup_train(self, rank, world_size, **kwargs):
        data_loader = kwargs.get("data_loader", None)

        self.teacher = self.get_model(is_teacher=True)

        self.student = self.get_model(is_teacher=False)
        self.student, self.teacher = self.student.to(self.device), self.teacher.to(self.device)

        if has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        if world_size > 1:
            self.student = DDP(self.student, device_ids=[rank])
            self.teacher = DDP(self.teacher, device_ids=[rank])

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.log(f"Student and Teacher are built: they are both {self.args.arch} network.")
        self.log(colorstr(f"Student model info"))
        model_info(self.student, detailed=False)
        self.log(colorstr(f"Teacher model info"))
        model_info(self.teacher, detailed=False)

        self.get_loss()
        self.build_optimizer()

        # for mixed precision training
        self.fp16_scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_fp16)

        # ============ init schedulers ... ============
        self.lr_schedule = cosine_scheduler(
            self.args.lr * (self.batch_size * world_size) / 256.,  # linear scaling rule
            self.args.min_lr,
            self.args.epochs, len(data_loader),
            warmup_epochs=self.args.warmup_epochs,
        )
        self.wd_schedule = cosine_scheduler(
            self.args.weight_decay,
            self.args.weight_decay_end,
            self.args.epochs, len(data_loader),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        self.momentum_schedule = cosine_scheduler(self.args.momentum_teacher, 1,
                                                  self.args.epochs, len(data_loader))

        self.log(f"{colorstr('Loss')} : {self.dino_loss.__class__.__name__}, "
                 f"{colorstr('optimizer')} : {self.optimizer.__class__.__name__} and schedulers ready.")

        to_restore = {"epoch": 0}
        restart_from_checkpoint(
            os.path.join(self.args.save_dir, "last.pth"),
            run_variables=to_restore,
            student=self.student,
            teacher=self.teacher,
            optimizer=self.optimizer,
            fp16_scaler=self.fp16_scaler,
            dino_loss=self.dino_loss,
        )
        self.start_epoch = to_restore["epoch"]
        self.log(f"Starting training of EsViT ! from epoch {self.start_epoch}")

    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)

        self.get_dataset()
        for train_set, test_set in self.trainset.generate_fold_dataset():
            train_set.transform = self.get_transform()
            self.train_loader, self.train_set = self.get_dataloader(dataset=train_set)
            self._setup_train(rank, world_size, data_loader=self.train_loader)

            self.epoch_time = None
            self.epoch_time_start = time.time()
            self.train_time_start = time.time()
            for epoch in range(self.start_epoch, self.args.epochs):
                self.teacher.eval(), self.student.train()
                self.epoch = epoch
                if rank != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                self._train_one_epoch(rank, world_size)

    def _train_one_epoch(self, rank=-1, world_size=1):
        pbar = enumerate(self.train_loader)
        nb = len(self.train_loader)  # number of batches
        if rank in (-1, 0):
            self.log(self.progress_string())
            pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)

        for it, batch in pbar:
            it = len(self.train_loader) * self.epoch + it  # global training iteration
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.wd_schedule[it]

            targets_mixup = None  # TODO: remove it
            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                batch = self.preprocess_batch(batch)
                teacher_output = self.teacher(batch['img'][:2])  # only the 2 global views pass through the teacher
                student_output = self.student(batch['img'])
                self.loss, self.loss_item = self.dino_loss(student_output, teacher_output, self.epoch, targets_mixup)

            # Backward
            # student update
            self.optimizer.zero_grad()
            param_norms = None
            self.fp16_scaler.scale(self.loss).backward()

            if self.args.clip_grad:
                param_norms = clip_gradients(self.student, self.args.clip_grad)
            cancel_gradients_last_layer(self.epoch, self.student, self.args.freeze_last_layer)
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()

            self.ema(it)

            torch.cuda.synchronize()
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
            losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
            if rank in (-1, 0):
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * loss_len) %
                    (f'{self.epoch + 1}/{self.epochs}', mem, *losses))

    def preprocess_batch(self, batch):
        batch['img'] = [im.cuda(non_blocking=True) for im in batch['img']]
        return batch

    def ema(self, it):
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(de_parallel(self.student), de_parallel(self.teacher)):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def progress_string(self):
        return ('\n' + '%11s' *
                (2 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names)
