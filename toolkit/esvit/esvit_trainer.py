import torchvision
import torch
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.engine.base_train import BaseTrainer
from toolkit.esvit import models
from toolkit.esvit.models import DINOHead
from toolkit.esvit.dino_loss import DDINOLoss
from toolkit.esvit.utils import get_params_groups, cosine_scheduler
from toolkit.esvit.augment import DataAugmentationLymphNode, FineTuneAugmentation
from toolkit import colorstr, TQDM_BAR_FORMAT
from toolkit.utils import RANK
from tqdm import tqdm


class EsVitTrainer(BaseTrainer):
    def __init__(self, cfg, overrides=None):
        super(EsVitTrainer, self).__init__(cfg=cfg, overrides=overrides)
        self.embed_dim = None
        self.loss = None
        self.dataset = self.get_dataset()
        self.student = self.get_model()
        self.teacher = self.get_model(is_teacher=True)

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

    def get_dataloader(self, dataset):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return data_loader

    def get_model(self, *args, **kwargs):
        model = None
        if self.args.arch in models.__dict__.keys():
            model = models.__dict__[self.args.arch](
                is_teacher=kwargs.get("is_teacher", False))
        else:
            self.log(f"Unknown architecture: {self.args.arch}")
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
        loss, loss_item = self.loss(preds['student_output'],
                                    preds['teacher_output'],
                                    preds['epoch'],
                                    preds['targets_mixup'])
        return loss, loss_item

    def _setup_train(self, rank, world_size, data_loader):

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        if world_size > 1:
            self.teacher = DDP(self.teacher, device_ids=[rank])
            self.student = DDP(self.student, device_ids=[rank])

        # Check AMP
        self.amp = torch.tensor(True).to(self.device)
        if RANK > -1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)

        self.loss_names = ["view-loss", "region-loss"]
        self.loss = DDINOLoss(
            self.args.out_dim,
            sum(self.args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
            self.args.warmup_teacher_temp,
            self.args.teacher_temp,
            self.args.warmup_teacher_temp_epochs,
            self.args.epochs,
        ).to(self.device)

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
        self.log(f"🚆 Loss-{self.loss.__class__.__name__} "
                 f"Optimizer-{self.optimizer.__class__.__name__} "
                 f"Schedulers-Cosine_scheduler 🚆\n")

    def build_optimizer(self, model, name, lr, momentum, decay):
        # ============ preparing optimizer ... ============
        params_groups = get_params_groups(model)
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        else:
            raise ValueError(f"Unknown optimizer {self.args.optimizer}")
        return optimizer

    def _do_train(self, rank=-1, world_size=1):
        for train_dataset, test_dataset in self.dataset.generate_fold_dataset():
            train_dataset.transform, test_dataset.transform = self.get_transform(is_backbone=True)
            self.train_loader = self.get_dataloader(train_dataset)
            self.test_loader = self.get_dataloader(test_dataset)

            if world_size > 1:
                self._setup_ddp(rank, world_size)

            self._setup_train(rank, world_size, self.train_loader)

            nb = len(self.train_loader)

            self.log(f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
                     f"Logging results to {colorstr('bold', self.save_dir)}\n"
                     f"Starting training for {self.epochs} epochs...")

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

                self.tloss = None
                self.optimizer.zero_grad()
                for i, batch in pbar:

                    # global iteration
                    ni = i + nb * epoch
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        param_group["lr"] = self.lr_schedule[ni]
                        if i == 0:  # only the first group is regularized
                            param_group["weight_decay"] = self.wd_schedule[ni]

                    # Forward
                    with torch.cuda.amp.autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        # only the 2 global views pass through the teacher
                        preds = {'teacher_output': self.teacher(batch['img'][:2]),
                                 'student_output': self.student(batch['img']),
                                 'epoch': epoch,
                                 'targets_mixup': None}
                        self.loss, self.loss_items = self.criterion(preds, batch)

                        if rank != -1:
                            self.loss *= world_size

                        self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                            else self.loss_items

                    # Backward
                    self.scaler.scale(self.loss).backward()

                    # Optimize
                    self.optimizer_step()

                    # Log
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                    losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)

                    if rank in (-1, 0):
                        pbar.set_description(
                            ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                            (f'{epoch + 1}/{self.epochs}', mem, *losses))

    def progress_string(self):
        return ('\n' + '%12s' * (2 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names)

    def preprocess_batch(self, batch):
        batch['img'] = [img.to(self.device, non_blocking=True).float() / 255 for img in batch['img']]
        return batch

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
