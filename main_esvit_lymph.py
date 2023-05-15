import argparse
import math

import time
import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np

import torch.backends.cuda

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.bulid_dataloader import create_loader

from toolkit.utils import (
    yaml_save, bool_flag, average_classification_reports,
    print_options, LOGGER, TQDM_BAR_FORMAT
)
from toolkit.utils.torch_utils import (
    init_seeds, de_parallel, has_batchnorms, cosine_scheduler,
    time_sync, load_pretrained_weights,
    restart_from_checkpoint, get_model_device,
    model_info, build_optimizer
)

from toolkit.utils.dist_utils import (
    init_distributed_mode, is_main_process,
    get_world_size, save_on_master
)

from toolkit.data.sampler import creat_sampler

from toolkit.utils.python_utils import merge_dict_with_prefix
from toolkit.utils.logger import MetricLogger
from toolkit.utils.loss import build_loss
from toolkit.utils.plots import show
from toolkit.models import create_teacher_student
from toolkit.data.augmentations import create_transform
from toolkit.utils.plots import plot_txt
from online_linear import run

from pathlib import Path
from copy import deepcopy


def get_args_parser():
    parser = argparse.ArgumentParser('EsViT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='swin_custom', type=str,
                        help="""Name of architecture to train. For quick experiments with ViTs,
                         we recommend using deit_tiny or deit_small.""")
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

    parser.add_argument('--use_mix_prediction', default=False, type=bool_flag,
                        help="Whether to use mix head in projection head (Default: False)")

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
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
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
    parser.add_argument('--local_crops_number', type=int, default=8,
                        help="""Number of small local views to generate. 
                        Set this parameter to 0 to disable multi-crop training. 
                        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1. """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")

    parser.add_argument('--global_crops_size', type=int, default=224)
    parser.add_argument('--local_crops_size', type=int, default=96, help="""Crop region size of local views to generate.
        When disabling multi-crop we recommend to use "--local_crops_size 96." """)

    # Augmentation parameters
    parser.add_argument('--aug_opt', type=str, default='lymph_node_aug', metavar='NAME',
                        help='Use different data augmentation policy. [deit_aug, dino_aug, mocov2_aug, basic_aug] \
                             "(default: dino_aug)')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')

    # Misc
    parser.add_argument('--data_path', default=['dataset', 'leaf_tumor_video'], type=str,
                        nargs='+', help='Please specify path to the ImageNet training data.')

    parser.add_argument('--pretrained_weights_ckpt', default='', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument('--save_dir', default="runs/debug", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('-- ', default=2, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--num_labels', default=2, type=int, help='number of classes in a dataset')
    parser.add_argument('--device', default="cuda")

    parser.add_argument('--weighted_sampler', default=True, type=bool_flag)

    # Linear Parser
    parser.add_argument('--linear_lr', type=float, default=0.01)
    parser.add_argument('--linear_epochs', type=int, default=1)

    return parser


def train_esvit(args):
    # Setting the distributed model
    init_distributed_mode(args)
    init_seeds(args.seed)
    if is_main_process():
        yaml_save(Path(args.save_dir) / 'args.yaml', data=vars(args))
        print_options(args)

    best_results, last_results = [], []

    device = torch.device(args.device)
    # ============ preparing data ... ============
    k_fold_dataset = KFoldLymphDataset(args.data_path, n_splits=5, shuffle=True, random_state=args.seed)

    for k, (train_set, test_set) in enumerate(k_fold_dataset.generate_fold_dataset()):
        # ============ K Folder training start  ... ============
        # Create folder output file
        args.fold_save_dir = Path(args.save_dir) / f'{k + 1}-fold'
        args.fold_save_dir.mkdir(parents=True, exist_ok=True)

        # transformation for backbone, train linear, test linear
        backbone_dataset = deepcopy(train_set)  # For DINO Backbone training
        backbone_dataset.transform = create_transform(args, args.aug_opt)

        # create sampler
        train_sampler, test_sampler = creat_sampler(args=args, train_dataset=train_set, test_dataset=test_set)

        data_loader = create_loader(args, args.batch_size_per_gpu, backbone_dataset, sampler=train_sampler)

        LOGGER.info(f"Backbone loaded : {len(backbone_dataset)} images.")
        LOGGER.info(f"Train loaded : {len(train_set)} images.")
        LOGGER.info(f"Val loaded : {len(test_set)} images.")

        # ============ Plotting Training val images ... ============
        if is_main_process():
            show(next(iter(backbone_dataset))['img'][:2], save_dir=args.fold_save_dir, name="global_view")
            show(next(iter(backbone_dataset))['img'][2:], save_dir=args.fold_save_dir, name="local_view")

        # ============ building student and teacher networks ... ============
        teacher, student = create_teacher_student(args)

        # Move networks to GPU
        student, teacher = student.to(device), teacher.to(device)

        # synchronize batch norms (if any)
        if args.distributed:
            if has_batchnorms(student):
                student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
                teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

            student = DDP(student, device_ids=[args.gpu])
            teacher = DDP(teacher, device_ids=[args.gpu])

        # teacher and student start with the same weights
        de_parallel(teacher).load_state_dict(de_parallel(student).state_dict())

        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False

        model_info(de_parallel(teacher), imgsz=224)
        model_info(de_parallel(student), imgsz=224)

        # ============ preparing loss ... ============
        criterion = build_loss(args, device)

        # ============ preparing optimizer ... ============
        optimizer = build_optimizer(optimizer=args.optimizer, model=student)

        # for mixed precision training
        scaler = amp.GradScaler() if args.amp else None

        # ============ init schedulers ... ============
        lr_schedule = cosine_scheduler(
            base_value=args.lr * (args.batch_size_per_gpu * get_world_size()) / 256.,  # linear scaling rule
            final_value=args.min_lr,
            epochs=args.epochs,
            niter_per_ep=len(data_loader),
            warmup_epochs=args.warmup_epochs,
        )
        wd_schedule = cosine_scheduler(
            base_value=args.weight_decay,
            final_value=args.weight_decay_end,
            epochs=args.epochs,
            niter_per_ep=len(data_loader),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = cosine_scheduler(
            base_value=args.momentum_teacher,
            final_value=1,
            epochs=args.epochs,
            niter_per_ep=len(data_loader))

        LOGGER.info(f"Loss, optimizer and schedulers ready.")

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
                **criterion,
            )
            LOGGER.info(f'Resumed from {args.pretrained_weights_ckpt}')

        restart_from_checkpoint(
            str(args.fold_save_dir / "last.pth"),
            run_variables=to_restore,
            student=de_parallel(student),
            teacher=de_parallel(teacher),
            optimizer=optimizer,
            scaler=scaler,
            **criterion,
        )
        start_epoch = to_restore["epoch"]

        LOGGER.info(f"Starting training of DINO! from epoch {start_epoch}")

        lowest_loss = sys.float_info.max
        best_w, last_w = args.fold_save_dir / f'best.pth', args.fold_save_dir / f'last.pth'
        txt_file = args.fold_save_dir / "log.txt"
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                data_loader.sampler.set_epoch(epoch)
            # ============ training one epoch of EsViT ... ============
            train_stats = train_one_epoch(
                student=student,
                teacher=teacher,
                criterion=criterion,
                data_loader=data_loader,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                wd_schedule=wd_schedule,
                momentum_schedule=momentum_schedule,
                epoch=epoch,
                scaler=scaler,
                args=args
            )

            # ============ writing logs ... ============
            save_dict = {
                'student': de_parallel(student).state_dict(),
                'teacher': de_parallel(teacher).state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
                **{f"{k}": v.state_dict() for k, v in criterion.items()}
            }
            if scaler:
                save_dict["scaler"] = scaler.state_dict()

            # Save the last ,best and delete weight
            save_on_master(save_dict, last_w)
            if train_stats["loss"] < lowest_loss:
                lowest_loss = train_stats["loss"]
                LOGGER.info(f"The lowest loss {lowest_loss:0.3f} model save path: {best_w}")
                save_on_master(save_dict, best_w)

            del save_dict

            log_stats = merge_dict_with_prefix({}, train_stats, "train_")
            log_stats["epoch"] = epoch

            for key in log_stats:
                if isinstance(log_stats[key], float):
                    log_stats[key] = "{:.6f}".format(round(log_stats[key], 6))

            if is_main_process():
                with txt_file.open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                plot_txt(txt_file, keyword="loss", save_dir=args.fold_save_dir, name="result")
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
        # ============ Prepare dataloader and dataset transform ============
        train_set.transform = create_transform(args, "eval_train")
        test_set.transform = create_transform(args, "eval_test")
        train_loader = create_loader(args, args.batch_size_per_gpu * 4, train_set, sampler=train_sampler)
        val_loader = create_loader(args, args.batch_size_per_gpu * 4, test_set, sampler=test_sampler)

        # plot image
        if is_main_process():
            show(next(iter(train_set))['img'], save_dir=args.fold_save_dir, name="train")
            show(next(iter(test_set))['img'], save_dir=args.fold_save_dir, name="test")

        load_pretrained_weights(
            model=de_parallel(teacher),
            pretrained_weights=best_w,
            checkpoint_key="teacher"
        )
        # Two times learning rate

        # Run the best pt.file
        best_result = run(
            train_loader=train_loader,
            val_loader=val_loader,
            model=de_parallel(teacher),
            args=args,
            save_dir=args.fold_save_dir / f'best_linear_eval',
            epochs=args.linear_epochs,
            lr=args.linear_lr
        )

        load_pretrained_weights(
            model=de_parallel(teacher),
            pretrained_weights=last_w,
            checkpoint_key="teacher"
        )

        # Run the last pt.file
        last_result = run(
            train_loader=train_loader,
            val_loader=val_loader,
            model=de_parallel(teacher),
            args=args,
            save_dir=args.fold_save_dir / f'last_linear_eval',
            epochs=args.linear_epochs,
            lr=args.linear_lr
        )

        # save linear eval result
        best_results.append(best_result)
        last_results.append(last_result)

        # save final yaml
        yaml_save(args.fold_save_dir / 'args.yaml', data=vars(args))

    if is_main_process():
        best_average = average_classification_reports(best_results)
        last_average = average_classification_reports(last_results)
        yaml_save(Path(args.save_dir) / 'best_average.yaml', data=best_average)
        yaml_save(Path(args.save_dir) / 'last_average.yaml', data=last_average)

    torch.cuda.empty_cache()


@torch.no_grad()
def get_features(val_loader, model, n, depths):
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
        criterion,
        data_loader,
        optimizer,
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        epoch,
        scaler,
        args,
):
    student.train(), teacher.eval()

    device = get_model_device(student)

    metric_logger = MetricLogger(delimiter=" ", save_dir=None)
    header = 'Epoch:[{}/{}]'.format(epoch, args.epochs)

    for it, batch in enumerate(metric_logger.log_every(data_loader, 20, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.to(device, non_blocking=True) for im in batch['img']]

        # teacher and student input
        teacher_input = images[:2]
        student_input = images

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(scaler is not None), torch.backends.cuda.sdp_kernel(enable_flash=False):
            with torch.no_grad():
                teacher_output = teacher(teacher_input)  # only the 2 global views pass through the teacher
            student_output = student(student_input)
            total_loss = torch.zeros(1, device=device)
            total_items = {}
            for loss_key, loss_fun in criterion.items():
                if loss_key == "ddino_loss":
                    loss, loss_items = loss_fun(
                        s_cls_out=student_output["head"],
                        s_region_out=student_output["dense_head"],
                        s_fea=student_output["output_fea"],
                        s_npatch=student_output["num_patch"],
                        t_cls_out=teacher_output["head"],
                        t_region_out=teacher_output["dense_head"],
                        t_fea=teacher_output["output_fea"],
                        t_npatch=teacher_output["num_patch"],
                        epoch=epoch
                    )
                    total_loss += loss
                    total_items = {**total_items, **loss_items}

                if loss_key == "dino_loss":
                    loss, loss_items = loss_fun(
                        student_output=student_output['head'],
                        teacher_output=teacher_output['head'],
                        epoch=epoch,
                        targets_mixup=None,
                    )
                    total_loss += loss
                    total_items = {**total_items, **loss_items}

                if loss_key == "mix_ddino_loss":
                    loss, loss_items = loss_fun(
                        s_mix_region_out=student_output["mix_head"],
                        s_fea=student_output["output_fea"],
                        s_npatch=student_output["num_patch"],
                        t_mix_region_out=teacher_output["mix_head"],
                        t_fea=teacher_output["output_fea"],
                        t_npatch=teacher_output["num_patch"],
                        epoch=epoch
                    )
                    total_loss += loss
                    total_items = {**total_items, **loss_items}

        if not math.isfinite(total_loss.item()):
            LOGGER.info("Loss is {}, stopping training".format(total_loss.item()))
            # ============ writing logs on a NaN for debug ... ============
            save_dict = {
                'student': de_parallel(student).state_dict(),
                'teacher': de_parallel(teacher).state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
                **{f"{k}": v.state_dict() for k, v in criterion.items()}
            }
            if scaler is not None:
                save_dict['scaler'] = scaler.state_dict()
            save_on_master(save_dict, args.save_dir / 'checkpoint_nan.pth')
            del save_dict
            torch.cuda.empty_cache()
            sys.exit(1)

        # student update
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.clip_grad is not None:
                # we should unscale the gradients of optimizes assigned params if you do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for name, param_q in de_parallel(student).named_parameters():
                param_k = de_parallel(teacher).state_dict()[name]
                param_k.mul_(m).add_((1 - m) * param_q.detach())

        # Logging
        time_sync()
        metric_logger.update(loss=sum([v for v in total_items.values()]))
        metric_logger.update(**total_items)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    LOGGER.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EsViT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_esvit(args)
