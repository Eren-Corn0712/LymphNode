import math
import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np

import torch.backends.cuda

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp

from toolkit.cfg import get_cfg

from toolkit.models import create_teacher_student

from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.bulid_dataloader import create_loader
from toolkit.data.sampler import creat_sampler
from toolkit.data.augmentations import create_transform

from toolkit.utils import (yaml_save, print_options, LOGGER)
from toolkit.utils.torch_utils import (init_seeds, de_parallel, has_batchnorms, cosine_scheduler, time_sync,
                                       restart_from_checkpoint, get_model_device, model_info, build_optimizer)

from toolkit.utils.dist_utils import (init_distributed_mode, is_main_process, get_world_size, save_on_master)
from toolkit.utils.files import (increment_path)
from toolkit.utils.python_utils import merge_dict_with_prefix
from toolkit.utils.logger import MetricLogger
from toolkit.utils.loss import build_loss
from toolkit.utils.plots import (show, plot_txt)
from pathlib import Path

import wandb


def main(args):
    # Setting the distributed training
    init_distributed_mode(args)
    init_seeds(args.seed)
    args.save_dir = Path(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok if is_main_process() else True))
    device = torch.device(args.device)

    if is_main_process():
        # save folder create
        args.save_dir.mkdir(parents=True, exist_ok=True)
        # save args
        yaml_save(args.save_dir / 'args.yaml', data=vars(args))
        print_options(args)

    # ============ preparing data ... ============
    k_fold_dataset = KFoldLymphDataset(args.data_path, n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for k, (train_set, test_set) in enumerate(k_fold_dataset.generate_fold_dataset()):
        # ============ K Folder training start  ... ============

        # ============ wandb run ========
        # Logger
        if args.wandb:
            # args.project always is runs/
            wandb.init(project=args.project, name=args.name, config=vars(args),
                       group=str(args.save_dir), job_type=f"{k + 1}_backbone",
                       dir=args.save_dir)

        # transformation for multi-crop
        train_set.transform = create_transform(args, args.aug_opt)

        # create sampler
        train_sampler, test_sampler = creat_sampler(args=args, train_dataset=train_set, test_dataset=test_set)

        data_loader = create_loader(args=args, dataset=train_set, sampler=train_sampler)

        LOGGER.info(f"Train loaded : {len(train_set)} images.")
        LOGGER.info(f"Val loaded : {len(test_set)} images.")

        # ============ Plotting Training val images ... ============
        fold_save_dir = args.save_dir / f'{k + 1}-fold'
        if is_main_process():
            # Create folder output file
            fold_save_dir.mkdir(parents=True, exist_ok=True)
            show(next(iter(train_set))['img'][:2], save_dir=fold_save_dir, name="global_view")
            show(next(iter(train_set))['img'][2:], save_dir=fold_save_dir, name="local_view")

        # ============ building student and teacher networks ... ============
        teacher, student = create_teacher_student(args)

        # Move networks to gpu
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
            str(fold_save_dir / "last.pth"),
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
        best_w, last_w = fold_save_dir / f'best.pth', fold_save_dir / f'last.pth'
        txt_file = fold_save_dir / "log.txt"

        for epoch in range(start_epoch, args.epochs):
            if args.distributed and hasattr(data_loader.sampler, "set_epoch"):
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

            if is_main_process():
                with txt_file.open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                # plot_txt(txt_file, keyword="loss", save_dir=fold_save_dir, name="result")

                if args.wandb:
                    wandb.log(log_stats)

        if args.wandb:
            wandb.finish()
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
        if it == 0:
            unique_label = torch.unique(batch['label'])
            count = {i.item(): torch.sum(torch.eq(batch['label'], i)).item() for i in unique_label}
            LOGGER.info(f"{count}")
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.to(device, non_blocking=True) for im in batch['img']]

        # teacher and student input
        if hasattr(args, "use_corr") and args.use_corr:
            teacher_input = images
            student_input = images
        else:
            teacher_input = images[:2]
            student_input = images

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(scaler is not None), torch.backends.cuda.sdp_kernel(enable_flash=False):
            student_output = student(student_input)
            with torch.no_grad():
                teacher_output = teacher(teacher_input)  # only the 2 global views pass through the teacher
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

                if loss_key == "trans_loss":
                    loss, loss_items = loss_fun(
                        s_trans_output=student_output["trans_head"],
                        s_fea=student_output["output_fea"],
                        s_npatch=student_output["num_patch"],
                        t_trans_output=teacher_output["trans_head"],
                        t_fea=teacher_output["output_fea"],
                        t_npatch=teacher_output["num_patch"],
                        epoch=epoch
                    )
                    total_loss += loss
                    total_items = {**total_items, **loss_items}
                if loss_key == "multi_level_loss":
                    loss, loss_items = loss_fun(
                        s_multi_level_region_out=student_output["multi_level"],
                        s_multi_level_fea=student_output["output_fea"],
                        s_multi_level_npatch=student_output["num_patch"],
                        t_multi_level_region_out=teacher_output["multi_level"],
                        t_multi_level_fea=teacher_output["output_fea"],
                        t_multi_level_npatch=teacher_output["num_patch"],
                        epoch=epoch
                    )
                    total_loss += loss
                    total_items = {**total_items, **loss_items}
                if loss_key == "cross_loss":
                    loss, loss_items = loss_fun(
                        teacher_output=teacher_output,
                        student_output=student_output,
                        epoch=epoch,
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
            return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
            de_teacher = de_parallel(teacher)
            de_student = de_parallel(student)
            for name, param_q in de_student.named_parameters():
                if name in de_teacher.state_dict():
                    param_k = de_teacher.state_dict()[name]
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


def esvit(cfg):
    args = get_cfg(cfg)
    main(args)


if __name__ == '__main__':
    pass
