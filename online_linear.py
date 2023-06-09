import toolkit.utils.logger
import toolkit.utils.torch_utils

import torch
import torch.nn as nn
import torch.distributed as dist
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score
import toolkit.models.swin_transformer as swin_transformer
import toolkit.models.resnet as resnet

from toolkit.utils import LOGGER
from toolkit.models.head import LinearClassifier
from toolkit.utils.torch_utils import de_parallel, time_sync, accuracy, detach_to_cpu_numpy, get_model_device
from toolkit.utils.dist_utils import save_on_master, is_main_process, get_world_size
from toolkit.utils.python_utils import merge_dict_with_prefix
from toolkit.utils.logger import (MetricLogger, SmoothedValue)


def run(
        train_loader,
        val_loader,
        model,
        args,
        save_dir,
        epochs=100,
        lr=0.001,
):
    if args.arch in swin_transformer.__dict__.keys():
        embed_dim = model.embed_dim
        depths = model.depths

        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        LOGGER.info(f"num_features: {num_features}")
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        LOGGER.info(f'num_features_linear {num_features_linear}')
        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)

    elif args.arch in resnet.__dict__.keys():
        depths = model.layers
        embed_dim = model.conv1.out_channels * model.layer1[-1].expansion
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        LOGGER.info(f"num_features: {num_features}")
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        LOGGER.info(f'num_features_linear {num_features_linear}')
        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)

    else:
        raise ValueError(f"We not implemented {args.arch}")

    model.cuda()
    model.eval()

    linear_classifier = linear_classifier.cuda()

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr * (train_loader.batch_size * get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    if save_dir and is_main_process():
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0
    best_f1 = 0
    best_result = None

    # last, best path
    last_w, best_w = (save_dir / "last.pth", save_dir / "best.pth")
    for epoch in range(0, epochs + 1):

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, depths)

        scheduler.step()

        log_stats = merge_dict_with_prefix({}, train_stats, "train_")

        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, depths)

        exclude = ("targets", "predicts")
        log_stats = merge_dict_with_prefix(log_stats, test_stats, "test_", exclude=exclude)

        LOGGER.info(f"Accuracy at epoch {epoch} test images: {test_stats['acc1']:.1f}%")

        if is_main_process():
            with (Path(save_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        save_dict = {
            "epoch": epoch,
            "state_dict": de_parallel(linear_classifier).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "best_f1": best_f1
        }

        save_on_master(save_dict, last_w)

        f1 = f1_score(test_stats['targets'], test_stats['predicts'], average='weighted')

        if is_main_process():
            if f1 > best_f1:
                name = train_loader.dataset.classes
                best_acc, best_f1 = test_stats["acc1"], f1
                LOGGER.info(f'Max accuracy so far: {best_acc:.4f}% F1-Score: {f1:.4f}')

                save_on_master(save_dict, best_w)

                cls_report = classification_report(
                    test_stats["targets"],
                    test_stats["predicts"],
                    target_names=name, output_dict=True)

                pd.DataFrame(cls_report).to_csv(save_dir / 'best.csv')
                best_result = cls_report

                cm = confusion_matrix(test_stats["targets"],
                                      test_stats["predicts"])

                cm = pd.DataFrame(cm, name, name)

                plt.figure(figsize=(9, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap='BuGn')
                plt.xlabel("prediction")
                plt.ylabel("label (ground truth)")
                plt.savefig(save_dir / "best_confusion_matrix.png")
                plt.close()

        del save_dict

    LOGGER.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    return best_result


def train(model, linear_classifier, optimizer, loader, epoch, n, depths):
    linear_classifier.train()
    model.eval()

    device = get_model_device(model)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for batch in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = batch['img'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.forward_return_n_last_blocks(inp, n, depths)

        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # Logging
        time_sync()
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    LOGGER.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def validate_network(val_loader, model, linear_classifier, n, depths):
    linear_classifier.eval()
    model.eval()

    device = get_model_device(model)

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    targets, predicts = [], []
    for batch in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = batch['img'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)

        # compute output
        output = model.forward_return_n_last_blocks(inp, n, depths)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        _, predict = torch.max(output.data, 1)

        targets.extend(detach_to_cpu_numpy(target))
        predicts.extend(detach_to_cpu_numpy(predict))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), num=batch_size)

    LOGGER.info("* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}"
                .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    states = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    states["targets"] = targets
    states["predicts"] = predicts
    return states
