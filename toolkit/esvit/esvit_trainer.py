import os
import time
from datetime import datetime

import torchvision
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist

from torch.utils.data import DataLoader, dataloader, distributed
from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.engine.base_train import BaseTrainer
from toolkit.esvit import models
from toolkit.esvit.models import DINOHead
from toolkit.esvit.dino_loss import DDINOLoss
from toolkit.esvit.utils import (get_params_groups, cosine_scheduler, has_batchnorms, LARS, clip_gradients,
                                 cancel_gradients_last_layer, restart_from_checkpoint)
from toolkit.esvit.augment import DataAugmentationLymphNode, FineTuneAugmentation
from toolkit import colorstr, TQDM_BAR_FORMAT
from toolkit.utils import RANK
from toolkit.data.utils import PIN_MEMORY
from toolkit.utils.torch_utils import de_parallel, deepcopy
from tqdm import tqdm


class EsVitTrainer(BaseTrainer):
    def __init__(self, cfg, overrides=None):
        super(EsVitTrainer, self).__init__(cfg=cfg, overrides=overrides)
        self.to_restore = None
        self.embed_dim = None
        self.dino_loss = None
        self.dataset = self.get_dataset()
        self.student = self.get_model()
        self.teacher = self.get_model(is_teacher=True)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.classifier = None

    def get_dataset(self):
        if self.args.k_fold:
            dataset = KFoldLymphDataset(self.args.data_path, self.args.k_fold)
        else:
            raise ValueError(f"{self.args.k_fold} Dataset not supported")
        return dataset

    def get_transform(self, is_backbone=False):
        train_transform, test_transform = None, None
        if is_backbone is True:
            if self.args.aug_opt == 'lymphNode':
                train_transform = DataAugmentationLymphNode(self.args.global_crops_scale,
                                                            self.args.local_crops_scale,
                                                            self.args.local_crops_number,
                                                            self.args.local_crops_size)

            return train_transform, None
        else:
            train_transform = FineTuneAugmentation(is_train=True)
            test_transform = FineTuneAugmentation(is_train=False)
        return train_transform, test_transform

    def get_dataloader(self, dataset, rank=-1, mode="train"):
        shuffle = mode == 'train'
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch,
            shuffle=shuffle and sampler is None,
            num_workers=self.args.num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=True,
            sampler=sampler,
            generator=generator,
        )
        self.log(f"Data loaded: there are {len(dataset)} {mode} images.")
        return data_loader

    def get_model(self, *args, **kwargs):
        model = None
        if self.args.arch in models.__dict__.keys():
            model = models.__dict__[self.args.arch](
                is_teacher=kwargs.get("is_teacher", False))
        else:
            self.log(f"Unknown architecture: {self.args.arch}")
        self.log(f"Success loading {self.args.arch} model")
        num_features = int(model.norm.normalized_shape[0])
        # Attach the new attributes
        model.use_dense_prediction = self.args.use_dense_prediction
        model.head = DINOHead(
            num_features,
            self.args.out_dim,
            use_bn=self.args.use_bn_in_head,
            norm_last_layer=self.args.norm_last_layer,
        )
        # Attach the new attributes
        if self.args.use_dense_prediction:
            model.head_dense = DINOHead(
                num_features,
                self.args.out_dim,
                use_bn=self.args.use_bn_in_head,
                norm_last_layer=self.args.norm_last_layer,
            )
        return model

    def criterion(self, preds, batch):
        """
        Returns loss and individual loss items as Tensor.
        """
        return self.dino_loss(preds['student_output'], preds['teacher_output'], preds['epoch'],
                              preds['targets_mixup'])

    def _setup_train(self, rank, world_size, data_loader):

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        if has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)

        if world_size > 1:
            self.teacher = DDP(self.teacher, device_ids=[rank])
            self.student = DDP(self.student, device_ids=[rank])

        # teacher and student start with the same weights
        de_parallel(self.teacher).load_state_dict(de_parallel(self.student).state_dict())

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.log(f"Student and Teacher are built: they are both {self.args.arch} network.")

        self.loss_names = ["view-loss", "region-loss"]

        # preparing loss
        if self.args.use_dense_prediction:
            self.dino_loss = DDINOLoss(
                self.args.out_dim,
                sum(self.args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
                self.args.warmup_teacher_temp,
                self.args.teacher_temp,
                self.args.warmup_teacher_temp_epochs,
                self.args.epochs,
            ).to(self.device)
        else:
            raise ValueError("Not support non-dense prediction.")

        self.optimizer = self.build_optimizer(model=self.student,
                                              name=self.args.optimizer,
                                              lr=None,
                                              momentum=None,
                                              decay=None)

        self.lr_schedule = cosine_scheduler(
            self.args.lr * (self.args.batch * world_size) / 256.,  # linear scaling rule
            self.args.min_lr,
            self.args.epochs, len(data_loader),
            warmup_epochs=self.args.warmup_epochs,
        )
        self.wd_schedule = cosine_scheduler(
            self.args.weight_decay,
            self.args.weight_decay_end,
            self.args.epochs, len(data_loader),
        )

        self.momentum_schedule = cosine_scheduler(self.args.momentum_teacher, 1,
                                                  self.args.epochs, len(data_loader))

        # for mixed precision training
        self.scaler = None
        if self.args.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_fp16)

        self.log(f"🚆 Loss-{self.dino_loss.__class__.__name__} "
                 f"Optimizer-{self.optimizer.__class__.__name__} "
                 f"Schedulers-Cosine_scheduler 🚆\n")

        self.restart_from_checkpoint()

    def restart_from_checkpoint(self):
        self.to_restore = {"epoch": 0}
        restart_from_checkpoint(
            str(self.last),
            run_variables=self.to_restore,
            student=self.student,
            teacher=self.teacher,
            optimizer=self.optimizer,
            fp16_scaler=self.scaler,
            dino_loss=self.dino_loss,
        )
        self.start_epoch = self.to_restore["epoch"]

    def build_optimizer(self, model, name, lr, momentum, decay):
        # preparing optimizer
        params_groups = get_params_groups(model)
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif self.args.optimizer == "lars":
            optimizer = LARS(params_groups)  # to use with convnet and large batches
        else:
            raise ValueError(f"Unknown optimizer {self.args.optimizer}")
        return optimizer

    def _do_train(self, rank=-1, world_size=1):
        if world_size > 1:
            self._setup_ddp(rank, world_size)

        for train_dataset, test_dataset in self.dataset.generate_fold_dataset():
            train_dataset.transform, test_dataset.transform = self.get_transform(is_backbone=True)
            self.train_loader = self.get_dataloader(train_dataset, rank, mode="train")
            self.test_loader = self.get_dataloader(test_dataset, rank, mode="test")

            self._setup_train(rank, world_size, self.train_loader)

            nb = len(self.train_loader)

            self.log(f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
                     f"Logging results to {colorstr('bold', self.save_dir)}\n"
                     f"Starting training for {self.epochs} epochs...")

            self.tloss = None
            self.optimizer.zero_grad()
            for epoch in range(self.start_epoch, self.epochs):
                self.epoch = epoch
                self.student.train()
                self.teacher.eval()

                if rank != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)

                if rank in (-1, 0):
                    self.log(self.progress_string())
                    pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)

                for i, batch in pbar:
                    # update weight decay and learning rate according to their schedule
                    it = len(self.train_loader) * epoch + i  # global training iteration
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group["lr"] = self.lr_schedule[it]
                        if i == 0:  # only the first group is regularized
                            param_group["weight_decay"] = self.wd_schedule[it]

                    # Forward
                    with torch.cuda.amp.autocast(self.scaler is not None):
                        batch = self.preprocess_batch(batch)
                        # only the 2 global views pass through the teacher
                        with torch.no_grad():
                            teacher_output = self.teacher(batch['img'][:2])
                        student_output = self.student(batch['img'])
                        preds = {'teacher_output': teacher_output,
                                 'student_output': student_output,
                                 'epoch': epoch,
                                 'targets_mixup': None}
                        self.loss, self.loss_items = self.criterion(preds, None)

                        if rank != -1:
                            self.loss *= world_size

                        self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                            else self.loss_items

                    # Backward
                    self.optimizer.zero_grad()
                    self.scaler.scale(self.loss).backward()
                    torch.cuda.synchronize()

                    # Optimize
                    self.optimizer_step()

                    # Log
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                    losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)

                    if rank in (-1, 0):
                        pbar.set_description(
                            ('%12s' * 2 + '%12.4g' * loss_len) %
                            (f'{epoch + 1}/{self.epochs}', mem, *losses))

                    # EMA update for the teacher
                    self.ema(it, de_parallel(self.student), de_parallel(self.teacher))

                    # Record lowest loss
                    self.fitness = torch.sum(self.tloss)
                    if not self.best_fitness or self.best_fitness > self.fitness:
                        self.best_fitness = self.fitness

                    # Save model
                    if rank in (-1, 0):
                        self.save_model()

    def progress_string(self):
        return ('\n' + '%12s' * (2 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names)

    def preprocess_batch(self, batch):
        batch['img'] = [img.to(self.device, non_blocking=True).float() / 255 for img in batch['img']]
        return batch

    def optimizer_step(self):
        if self.args.clip_grad:
            self.scaler.unscale_(self.optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def save_model(self):
        save_dict = {
            'student': self.student.state_dict(),
            'teacher': self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch + 1,
            'args': self.args,
            'dino_loss': self.dino_loss.state_dict(),
            'fp16_scaler': self.scaler.state_dict()
        }
        torch.save(save_dict, self.last)

        if self.best_fitness == self.fitness:
            torch.save(save_dict, self.best)

    def ema(self, it, student, teacher):
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
