import datetime
import json
import os
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data
import torchvision
import torch.distributed
import numpy as np
from pathlib import Path

from torch import nn
from torch.utils.data.dataloader import default_collate

from toolkit.utils import yaml_save, print_options, average_classification_reports
from toolkit.utils.logger import (MetricLogger, SmoothedValue)
from toolkit.utils.dist_utils import (reduce_across_processes, is_main_process, save_on_master, init_distributed_mode)
from toolkit.utils.files import mkdir
from toolkit.utils.torch_utils import (accuracy, ExponentialMovingAverage, set_weight_decay, init_seeds,
                                       detach_to_cpu_numpy)
from toolkit.utils.python_utils import merge_dict_with_prefix, batch_dataconcat
from toolkit.data.augmentations import RandomMixup, RandomCutmix
from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.augmentations import create_transform
from toolkit.utils.plots import plot_confusion_matrix, plot_txt
from toolkit.data.sampler import creat_sampler
from eval_linear import case_analysis
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = batch['img'].to(device), batch['label'].to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), num=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    save_dict = {}
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        image = batch['img'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)
        output = model(image)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        _, predict = torch.max(output.data, 1)

        save_dict = batch_dataconcat(
            save_dict,
            dict(
                type_name=batch['type_name'],
                label=detach_to_cpu_numpy(batch['label'].view(-1)),
                patient_id=batch['patient_id'],
                im_file=batch['im_file'],
                predict=detach_to_cpu_numpy(predict.view(-1)),
            )
        )

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), num=batch_size)
        # metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        num_processed_samples += batch_size

    # gather the stats from all processes

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f}")

    states = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    states['save_dict'] = save_dict
    return states


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def main(args):
    if args.save_dir:
        mkdir(args.save_dir)

    init_distributed_mode(args)
    init_seeds(seed=0, deterministic=args.use_deterministic_algorithms)
    print_options(args)
    yaml_save(Path(args.save_dir) / "args.yaml", vars(args))

    device = torch.device(args.device)
    best_results = []
    k_fold_dataset = KFoldLymphDataset(args.data_path, n_splits=5, shuffle=True, random_state=0)
    print(f"Total data : {len(k_fold_dataset)}")
    for k, (train_set, test_set) in enumerate(k_fold_dataset.generate_fold_dataset()):
        # create the folder output
        print(f"train data : {len(train_set)}, test_set : {len(test_set)}")

        fold_save_dir = Path(args.save_dir) / f'{k + 1}-fold'
        fold_save_dir.mkdir(parents=True, exist_ok=True)

        train_set.transform = create_transform(args, "preset_train")
        test_set.transform = create_transform(args, "preset_test")

        train_sampler, test_sampler = creat_sampler(args, train_dataset=train_set, test_dataset=test_set)

        collate_fn = None
        num_classes = len(train_set.classes)
        mixup_transforms = []
        if args.mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
        if args.cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))

        data_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        data_loader_test = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True
        )

        print("Creating model")
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
        model.to(device)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        custom_keys_weight_decay = []
        if args.bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
        if args.transformer_embedding_decay is not None:
            for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
                custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
        parameters = set_weight_decay(
            model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )

        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

        scaler = torch.cuda.amp.GradScaler() if args.amp else None

        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                                                gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        model_ema = None
        if args.model_ema:
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])

        if args.test_only:
            # We disable the cudnn benchmarking because it can noticeably affect the accuracy
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if model_ema:
                evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            else:
                evaluate(model, criterion, data_loader_test, device=device)
            return

        print("Start training")
        start_time = time.time()
        best_acc, best_f1, best_result = 0, 0, None

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_stats = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema,
                                          scaler)

            log_stats = merge_dict_with_prefix({}, train_stats, "train_")

            lr_scheduler.step()
            test_stats = evaluate(model, criterion, data_loader_test, device=device)

            exclude = ("save_dict",)
            log_stats = merge_dict_with_prefix(log_stats, test_stats, "test_", exclude=exclude)
            log_stats['epoch'] = epoch

            if is_main_process():
                with (Path(fold_save_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if model_ema:
                evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()

            save_on_master(checkpoint, fold_save_dir / "last.pth")

            if is_main_process():
                label = test_stats['save_dict']['label']
                predict = test_stats['save_dict']['predict']
                f1 = f1_score(label, predict, average='weighted')
                if f1 > best_f1:
                    name = train_set.classes
                    best_acc, best_f1 = test_stats["acc1"], f1
                    print(f'Max accuracy so far: {best_acc:.4f}% F1-Score: {f1:.4f}')

                    save_on_master(checkpoint, fold_save_dir / "best.pth")

                    cls_report = classification_report(
                        label,
                        predict,
                        labels=np.arange(0, len(name)),
                        target_names=name,
                        output_dict=True)

                    # save to csv
                    pd.DataFrame(cls_report).to_csv(fold_save_dir / 'best.csv')
                    best_result = cls_report

                    cm = confusion_matrix(label, predict)
                    plot_confusion_matrix(cm, name, fold_save_dir)
                    case_analysis(test_stats['save_dict'], fold_save_dir)

        best_results.append(best_result)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

    if is_main_process():
        best_average = average_classification_reports(best_results)
        yaml_save(Path(args.save_dir) / 'best_average.yaml', data=best_average)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default=["dataset_clean"], help="dataset path", nargs='+')
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda:0", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=1e-6, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--save-dir", default="runs/20230531_suptrain", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=224, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
