import csv

import toolkit.utils.logger
import toolkit.utils.torch_utils

import torch
import torch.nn as nn
import json
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import toolkit.models.swin_transformer as swin_transformer
import toolkit.models.resnet as resnet
from toolkit.utils import LOGGER
from toolkit.models import create_linear_layer
from toolkit.utils.torch_utils import (de_parallel, time_sync, accuracy, detach_to_cpu_numpy, get_model_device,
                                       restart_from_checkpoint)
from toolkit.utils.dist_utils import save_on_master, is_main_process, get_world_size
from toolkit.utils.python_utils import (merge_dict_with_prefix, batch_dataconcat)
from toolkit.utils.logger import (MetricLogger, SmoothedValue)
from toolkit.utils.plots import plot_txt, plot_confusion_matrix



def run(
        train_loader,
        val_loader,
        model,
        args,
        save_dir,
        epochs=100,
        lr=0.001,
):
    linear_classifier = create_linear_layer(model, args)

    device = get_model_device(model)
    model.eval()

    linear_classifier = linear_classifier.to(device=device)
    if args.distributed:
        # model = DDP(model, device_ids=[args.gpu])
        linear_classifier = DDP(linear_classifier, device_ids=[args.gpu])

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

    to_restore = {"epoch": 0, "best_acc": 0.}
    restart_from_checkpoint(
        save_dir / "last.pth",
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    best_f1 = 0
    best_result = None

    # last, best ckpt path
    last_w, best_w = (save_dir / "last.pth", save_dir / "best.pth")
    txt_file = save_dir / "log.txt"
    for epoch in range(start_epoch, epochs + 1):
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train(
            model=model,
            linear_classifier=linear_classifier,
            optimizer=optimizer,
            loader=train_loader,
            epoch=epoch,
            n=args.n_last_blocks,
            depths=depths
        )

        scheduler.step()

        log_stats = merge_dict_with_prefix({}, train_stats, "train_")

        test_stats = validate_network(
            val_loader=val_loader,
            model=model,
            linear_classifier=linear_classifier,
            n=args.n_last_blocks,
            depths=depths)

        exclude = ("save_dict",)
        log_stats = merge_dict_with_prefix(log_stats, test_stats, "test_", exclude=exclude)
        log_stats['epoch'] = epoch

        LOGGER.info(f"Accuracy at epoch {epoch} test images: {test_stats['acc1']:.1f}%")

        for key in log_stats:
            if isinstance(log_stats[key], float):
                log_stats[key] = "{:.6f}".format(round(log_stats[key], 6))

        if is_main_process():
            with txt_file.open("a") as f:
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

        if is_main_process():
            f1 = f1_score(test_stats['save_dict']['label'],
                          test_stats['save_dict']['predict'], average='weighted')
            if f1 > best_f1:
                name = train_loader.dataset.classes
                best_acc, best_f1 = test_stats["acc1"], f1
                LOGGER.info(f'Max accuracy so far: {best_acc:.4f}% F1-Score: {f1:.4f}')

                save_on_master(save_dict, best_w)

                cls_report = classification_report(
                    test_stats['save_dict']['label'],
                    test_stats['save_dict']['predict'],
                    labels=np.arange(0, len(name)),
                    target_names=name,
                    output_dict=True)

                pd.DataFrame(cls_report).to_csv(save_dir / 'best.csv')
                best_result = cls_report

                cm = confusion_matrix(test_stats['save_dict']['label'],
                                      test_stats['save_dict']['predict'])
                plot_confusion_matrix(cm, name, save_dir)
                case_analysis(test_stats['save_dict'], save_dir)

        del save_dict

    # Plot loss
    if is_main_process():
        plot_txt(txt_file, keyword="acc", save_dir=save_dir, name="result")

    LOGGER.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))

    return best_result


def train(model, linear_classifier, optimizer, loader, epoch, n, depths):
    linear_classifier.train()
    model.eval()

    device = get_model_device(model)  # get device

    metric_logger = MetricLogger(delimiter=" ")
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
        batch_size = inp.shape[0]
        time_sync()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), num=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    LOGGER.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def validate_network(val_loader, model, linear_classifier, n, depths):
    linear_classifier.eval()
    model.eval()

    device = get_model_device(model)

    metric_logger = MetricLogger(delimiter=" ")
    header = 'Test:'
    save_dict: dict = {}
    for batch in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = batch['img'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)

        # compute output
        backbone_output = model.forward_return_n_last_blocks(inp, n, depths)
        output = linear_classifier(backbone_output)
        loss = nn.CrossEntropyLoss()(output, target)

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

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), num=batch_size)

    LOGGER.info("* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}"
                .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    states = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    states['save_dict'] = save_dict
    return states


def case_analysis(data, save_dir):
    output = []
    patient_id = np.array(data['patient_id'])
    type_name = np.array(data['type_name'])
    unique_id = np.unique(patient_id)
    im_file = np.array(data['im_file'])

    for p_id in unique_id:
        index = np.where(patient_id == p_id)
        type_n = np.unique(type_name[index])
        im_f = np.unique(im_file[index])
        if type_n.size != 1:
            raise ValueError(f"The id {p_id} have multi label!")

        label = data['label'][index]
        predict = data['predict'][index]
        curr_num = np.count_nonzero((label == predict))
        acc = curr_num / len(label) * 100

        d = dict(
            file=im_f[0],
            id=p_id,
            type=type_n[0],
            label_num=len(label),
            current_num=int(curr_num),
            accuracy=f"{acc:.4f}"
        )
        output.append(d)

    if save_dir:
        csv_file = save_dir / "analysis.csv"
        fieldnames = output[0].keys()
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, restval=' ', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(output)
