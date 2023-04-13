import argparse
import datetime
import math

import time
import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
import torchvision
import toolkit.models.swin_transformer as swin_transformer
import toolkit.models.resnet as resnet

from pathlib import Path
from copy import deepcopy

from toolkit.data.lymph_dataset import KFoldLymphDataset

from toolkit.utils import yaml_load, yaml_save, yaml_print, bool_flag
from toolkit.utils.torch_utils import (init_seeds, de_parallel, has_batchnorms, cosine_scheduler,
                                       time_sync, get_params_groups, LARS, MultiCropWrapper, load_pretrained_weights,
                                       restart_from_checkpoint, clip_gradients, cancel_gradients_last_layer,
                                       model_info, select_device, build_optimizer)

from toolkit.utils.dist_utils import (init_distributed_mode, is_main_process,
                                      get_world_size, save_on_master)

from toolkit.utils.logger import MetricLogger
from toolkit.utils.loss import DDINOLoss, DINOLoss
from toolkit.utils.plots import show
from toolkit.models.head import DINOHead
from toolkit.data.bulid_dataloader import build_dataloader
from toolkit.data.augment import get_transform
from torchvision.utils import make_grid
from online_linear import run


def get_args_parser():
    parser = argparse.ArgumentParser('EsViT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='resnet50', type=str,
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (deit_tiny, deit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with deit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--use_dense_prediction', default=True, type=bool_flag,
                        help="Whether to use dense prediction in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--amp', type=bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, nargs='+', default=(8,), help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--local_crops_size', type=int, nargs='+', default=(96,), help="""Crop region size of local views to generate.
        When disabling multi-crop we recommend to use "--local_crops_size 96." """)

    # Augmentation parameters
    parser.add_argument('--aug-opt', type=str, default='lymph_node_aug', metavar='NAME',
                        help='Use different data augmentation policy. [deit_aug, dino_aug, mocov2_aug, basic_aug] \
                             "(default: dino_aug)')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset
    parser.add_argument('--dataset', default="imagenet1k", type=str, help='Pre-training dataset.')
    parser.add_argument('--zip_mode', type=bool_flag, default=False,
                        help="""Whether or not to use zip file.""")
    parser.add_argument('--tsv_mode', type=bool_flag, default=False,
                        help="""Whether or not to use tsv file.""")
    parser.add_argument('--sampler', default="distributed", type=str, help='Sampler for dataloader.')

    # Misc
    parser.add_argument('--data_path', default=['dataset', 'leaf_tumor_video'], type=str,
                        nargs='+', help='Please specify path to the ImageNet training data.')

    parser.add_argument('--pretrained_weights_ckpt', default='', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument('--output_dir', default="runs/debug", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for DeiT-Small and to True with ViT-Base.""")
    parser.add_argument('--num_labels', default=2, type=int, help='number of classes in a dataset')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--device', default="cuda")
    return parser


def train_esvit(args):
    # Setting the distributed model
    init_distributed_mode(args)
    init_seeds(args.seed)

    yaml_save(Path(args.output_dir) / 'args.yaml', data=vars(args))
    yaml_print(Path(args.output_dir) / 'args.yaml')

    device = torch.device(args.device)
    # ============ preparing data ... ============
    k_fold_dataset = KFoldLymphDataset(args.data_path)
    for k, (train_set, test_set) in enumerate(k_fold_dataset.generate_fold_dataset()):
        # Create folder output file
        fold_output_dir = Path(args.output_dir) / f'{k + 1}-fold'
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # transformation for backbone, train linear, test linear
        backbone_dataset = deepcopy(train_set)
        train_set.transform = get_transform(args, "eval_train")
        test_set.transform = get_transform(args, "eval_test")
        backbone_dataset.transform = get_transform(args, args.aug_opt)

        data_loader, train_loader, val_loader = build_dataloader(args, backbone_dataset, train_set, test_set)

        views, train_imgs, val_imgs = next(iter(data_loader))['img'], next(iter(train_loader))['img'], next(
            iter(val_loader))['img']

        if is_main_process():
            show_num = min(4, args.batch_size_per_gpu)
            for i in range(show_num):
                global_view = [view[i] for view in views[:2]]
                local_view = [view[i] for view in views[2:]]
                show(make_grid(global_view), fold_output_dir, f'{i}-global-view')
                show(make_grid(local_view), fold_output_dir, f'{i}-local-view')
            show(make_grid(train_imgs[:show_num]), fold_output_dir, f'train-imgs')
            show(make_grid(val_imgs[:show_num]), fold_output_dir, f'val-imgs')
            del views, train_imgs, val_imgs

        # ============ building student and teacher networks ... ============

        # if the network is a 4-stage vision transformer (i.e. swin)
        if args.arch in swin_transformer.__dict__.keys():
            student = swin_transformer.__dict__[args.arch]()
            teacher = swin_transformer.__dict__[args.arch](is_teacher=True)
            student.head = DINOHead(
                student.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

            setattr(student, "use_dense_prediction", args.use_dense_prediction)
            setattr(teacher, "use_dense_prediction", args.use_dense_prediction)
            if args.use_dense_prediction:
                student.head_dense = DINOHead(
                    student.num_features,
                    args.out_dim,
                    use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer,
                )
                teacher.head_dense = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)


        # otherwise, we check if the architecture is in torchvision models
        elif args.arch in resnet.__dict__.keys():
            student = resnet.__dict__[args.arch]()
            teacher = resnet.__dict__[args.arch]()
            student.head = DINOHead(
                student.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            teacher.head = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

            setattr(student, "use_dense_prediction", args.use_dense_prediction)
            setattr(teacher, "use_dense_prediction", args.use_dense_prediction)
            if args.use_dense_prediction:
                student.head_dense = DINOHead(
                    student.num_features,
                    args.out_dim,
                    use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer,
                )
                teacher.head_dense = DINOHead(teacher.num_features, args.out_dim, args.use_bn_in_head)

        else:
            raise ValueError(f"Unknow architecture: {args.arch}")

        # move networks to gpu
        student, teacher = student.to(device), teacher.to(device)

        # synchronize batch norms (if any)
        if args.distributed:
            if has_batchnorms(student):
                student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
                teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

        # teacher and student start with the same weights
        de_parallel(teacher).load_state_dict(de_parallel(student).state_dict())

        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False

        print(f"Student and Teacher are built: they are both {args.arch} network.")

        model_info(de_parallel(teacher), imgsz=224)
        model_info(de_parallel(student), imgsz=224)

        # ============ preparing loss ... ============
        if args.use_dense_prediction:
            # Both view and region level tasks are considered
            dino_loss = DDINOLoss(
                args.out_dim,
                sum(args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
                args.warmup_teacher_temp,
                args.teacher_temp,
                args.warmup_teacher_temp_epochs,
                args.epochs,
            ).to(device)
        else:
            # Only view level task is considered
            dino_loss = DINOLoss(
                args.out_dim,
                sum(args.local_crops_number) + 2,  # total number of crops = 2 global crops + local_crops_number
                args.warmup_teacher_temp,
                args.teacher_temp,
                args.warmup_teacher_temp_epochs,
                args.epochs,
            ).to(device)

        # ============ preparing optimizer ... ============
        optimizer = build_optimizer(optimizer=args.optimizer, model=student)

        # for mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # ============ init schedulers ... ============
        lr_schedule = cosine_scheduler(
            args.lr * (args.batch_size_per_gpu * get_world_size()) / 256.,  # linear scaling rule
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs,
        )
        wd_schedule = cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            args.epochs, len(data_loader),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                             args.epochs, len(data_loader))

        print(f"Loss, optimizer and schedulers ready.")

        # ============ optionally resume training ... ============
        to_restore = {"epoch": 0}

        if args.pretrained_weights_ckpt:
            restart_from_checkpoint(
                os.path.join(args.pretrained_weights_ckpt),
                run_variables=to_restore,
                student=de_parallel(student),
                teacher=de_parallel(teacher),
                optimizer=optimizer,
                scaler=scaler,
                dino_loss=dino_loss,
            )
            print(f'Resumed from {args.pretrained_weights_ckpt}')

        restart_from_checkpoint(
            str(fold_output_dir / "last.pth"),
            run_variables=to_restore,
            student=de_parallel(student),
            teacher=de_parallel(teacher),
            optimizer=optimizer,
            scaler=scaler,
            dino_loss=dino_loss,
        )
        start_epoch = to_restore["epoch"]

        start_time = time.time()
        print(f"Starting training of DINO! from epoch {start_epoch}")
        lowest_loss = sys.float_info.max
        best_w = Path(fold_output_dir) / f'best.pth'
        last_w = Path(fold_output_dir) / f'last.pth'
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                data_loader.sampler.set_epoch(epoch)
            # ============ training one epoch of EsViT ... ============
            train_stats = train_one_epoch(
                student=student,
                teacher=teacher,
                dino_loss=dino_loss,
                data_loader=data_loader,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                wd_schedule=wd_schedule,
                momentum_schedule=momentum_schedule,
                epoch=epoch,
                scaler=scaler,
                args=args)

            # ============ writing logs ... ============
            save_dict = {
                'student': de_parallel(student).state_dict(),
                'teacher': de_parallel(teacher).state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
                'dino_loss': dino_loss.state_dict(),
            }
            if scaler is not None:
                save_dict['scaler'] = scaler.state_dict()

            # Save the last ,best and delete weight
            save_on_master(save_dict, last_w)

            if train_stats["loss"] < lowest_loss:
                lowest_loss = train_stats["loss"]
                print(f"The lowest loss {lowest_loss} model save on master {best_w}")
                save_on_master(save_dict, best_w)

            del save_dict

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}

            if is_main_process():
                with (Path(fold_output_dir / "log.txt")).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # features, labels = get_features(val_loader,
            #                                 teacher,
            #                                 args.n_last_blocks,
            #                                 args.avgpool_patchtokens,
            #                                 config.MODEL.SPEC['DEPTHS'])
            #
            # tsne = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(features)
            # tsne_plot(tsne, labels, fold_output_dir, epoch)
            # tsne_video(fold_output_dir, fold_output_dir)
            # del tsne, features, labels, save_dict, train_stats, log_stats,

        # ============ Linear evaluation ============

        load_pretrained_weights(model=de_parallel(teacher),
                                pretrained_weights=best_w,
                                checkpoint_key="teacher")
        # Run the best pt.file
        run(train_loader=train_loader,
            val_loader=val_loader,
            model=de_parallel(teacher),
            args=args,
            save_dir=fold_output_dir / f'best_linear',
            epochs=100)

        load_pretrained_weights(model=de_parallel(teacher),
                                pretrained_weights=last_w,
                                checkpoint_key="teacher")

        # Run the last pt.file
        run(train_loader=train_loader,
            val_loader=val_loader,
            model=de_parallel(teacher),
            args=args,
            save_dir=fold_output_dir / f'last_linear',
            epochs=100)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        torch.cuda.empty_cache()
        print('Training time {}'.format(total_time_str))


@torch.no_grad()
def get_features(val_loader, model, n, avgpool, depths):
    # we'll store the features as NumPy array of size num_images x feature_size
    metric_logger = MetricLogger(delimiter="  ")

    features = None
    labels = []
    header = 'Get Features'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model.forward_return_n_last_blocks(inp, n, depths)

        labels.extend(target.tolist())
        if features is not None:
            features = np.concatenate((features, output.detach().cpu().numpy()))
        else:
            features = output.detach().cpu().numpy()

    return features, labels


def train_one_epoch(
        student,
        teacher,
        dino_loss,
        data_loader,
        optimizer,
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        epoch,
        scaler,
        args):
    student.train(), teacher.eval()
    device = next(student.parameters()).device  # get model device

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch['img']

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.to(device, non_blocking=True) for im in images]

        # teacher and student input
        teacher_input = images[:2]
        student_input = images

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(scaler is not None):
            with torch.no_grad():
                teacher_output = teacher(teacher_input)  # only the 2 global views pass through the teacher
            student_output = student(student_input)
            loss, loss_items = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            # ============ writing logs on a NaN for debug ... ============
            save_dict = {
                'student': de_parallel(student).state_dict(),
                'teacher': de_parallel(teacher).state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
                'dino_loss': dino_loss.state_dict(),
            }
            if scaler is not None:
                save_dict['scaler'] = scaler.state_dict()
            save_on_master(save_dict, args.output_dir / 'checkpoint_nan.pth')
            del save_dict
            torch.cuda.empty_cache()
            sys.exit(1)

        # student update
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)  # clip gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for name, param_q in de_parallel(student).named_parameters():
                param_k = de_parallel(teacher).state_dict()[name]
                param_k.mul_(m).add_((1 - m) * param_q.detach())

        # logging
        time_sync()
        metric_logger.update(loss=loss_items.sum())
        metric_logger.update(global_loss=loss_items[0])
        metric_logger.update(local_loss=loss_items[1])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EsViT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_esvit(args)