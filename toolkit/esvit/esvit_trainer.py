import torchvision
import torch
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.engine.base_train import BaseTrainer
from toolkit.esvit.model import DINOHead
from toolkit.esvit.dino_loss import DDINOLoss
from toolkit.esvit.utils import get_params_groups, cosine_scheduler
from toolkit.esvit.augment import DataAugmentationLymphNode, FineTuneAugmentation


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
        if self.args.arch in torchvision.models.__dict__.keys():
            if kwargs.get('is_teacher', False):
                model = torchvision.models.__dict__[self.args.arch](dropout=0.0)
            else:
                model = torchvision.models.__dict__[self.args.arch](dropout=0.1)
        else:
            self.log(f"Unknown architecture: {self.args.arch}")
        num_features = int(model.norm.normalized_shape[0])

        # Attach the new attributes
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

        return None

    def _setup_train(self, rank, world_size, data_loader):
        if world_size > 1:
            self.teacher = DDP(self.teacher, device_ids=[rank])
            self.student = DDP(self.student, device_ids=[rank])

        self.loss = DDINOLoss(
            self.args.out_dim,
            self.args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
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
            train_loader = self.get_dataloader(train_dataset)
            test_loader = self.get_dataloader(test_dataset)
            self._setup_train(rank, world_size, train_loader)
