import torch
import torch.nn as nn
import argparse
import json
import csv
import numpy as np
import pandas as pd

from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP

from toolkit.cfg import get_cfg

from toolkit.data.bulid_dataloader import create_loader
from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.data.sampler import creat_sampler
from toolkit.data import create_transform

from toolkit.utils.dist_utils import (
    init_distributed_mode, is_main_process,
    get_world_size, save_on_master
)

from toolkit.utils.torch_utils import (init_seeds, de_parallel, time_sync, load_pretrained_weights,
                                       restart_from_checkpoint, get_model_device, accuracy,
                                       detach_to_cpu_numpy)
from toolkit.utils.python_utils import (merge_dict_with_prefix, batch_dataconcat)

from toolkit.models import create_teacher_student, create_linear_layer
from toolkit.utils import (LOGGER, yaml_save, print_options, average_classification_reports)
from toolkit.utils.plots import (show, plot_txt, plot_confusion_matrix)
from toolkit.utils.logger import (MetricLogger, SmoothedValue)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from toolkit.utils.files import (increment_path)

import wandb


def main(args):
    init_distributed_mode(args)
    init_seeds(args.seed)
    args.save_dir = Path(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok if is_main_process() else True))
    device = torch.device(args.device)

    if is_main_process():
        # save folder create
        args.save_dir.mkdir(parents=True, exist_ok=True)
        # save args
        yaml_save(args.save_dir / f'args_{args.exp_name}.yaml', data=vars(args))
        print_options(args)

    best_results = []
    k_fold_dataset = KFoldLymphDataset(args.data_path, n_splits=5, shuffle=True, random_state=args.seed)

    if args.split == "default":
        fold = k_fold_dataset.generate_fold_dataset
    else:
        fold = k_fold_dataset.generate_patient_fold_dataset
    LOGGER.info(f"Folder Split by {fold.__name__}")

    for k, (train_set, test_set) in enumerate(fold()):
        # ============ K Folder training start  ... ============

        # ============ wandb run ========
        # Logger
        if args.wandb:
            # args.project always is runs/
            wandb.init(project=args.project, name=args.exp_name, config=vars(args),
                       group=str(args.save_dir), job_type=f"{k + 1}_eval",
                       dir=args.save_dir)

        if args.aug_opt == "eval":
            train_set.transform = create_transform(args, "eval_train")
            test_set.transform = create_transform(args, "eval_test")
        elif args.aug_opt == "eval_norm":
            train_set.transform = create_transform(args, "eval_train_norm")
            test_set.transform = create_transform(args, "eval_test_norm")
        else:
            raise ValueError(f"Augmentation key {args.aug_opt} is not support!")

        LOGGER.info(f"Train loaded : {len(train_set)} images.")
        LOGGER.info(f"Val loaded : {len(test_set)} images.")

        fold_save_dir = args.save_dir / f'{k + 1}-fold'
        # plot image
        if is_main_process():
            fold_save_dir.mkdir(parents=True, exist_ok=True)
            show(next(iter(train_set))['img'], save_dir=fold_save_dir, name="train")
            show(next(iter(test_set))['img'], save_dir=fold_save_dir, name="test")

        train_sampler, test_sampler = creat_sampler(args=args, train_dataset=train_set, test_dataset=test_set)

        train_loader = create_loader(args=args, dataset=train_set, sampler=train_sampler)
        val_loader = create_loader(args=args, dataset=test_set, sampler=test_sampler)

        # ============ building backbone networks ... ============
        model, _ = create_teacher_student(args)  # only one model need to be evaluated
        model = model.to(device)
        model.train() if args.fine_tune else model.eval()

        # load weights to evaluate
        pretrained_weights = fold_save_dir / f'{args.pretrained_weights_key}.pth'
        load_pretrained_weights(model, pretrained_weights, checkpoint_key=args.checkpoint_key)

        # ============ building linear networks ... ============
        linear_classifier = create_linear_layer(model, args)
        linear_classifier = linear_classifier.to(device=device)
        linear_classifier.train()

        if args.distributed:
            # model = DDP(model, device_ids=[args.gpu])
            linear_classifier = DDP(linear_classifier, device_ids=[args.gpu])

        # set optimizer
        param_groups = [{'params': linear_classifier.parameters(),
                         'lr': args.lr * (args.batch_size_per_gpu * get_world_size()) / 256.}]

        if args.fine_tune:
            param_groups.append({'params': model.parameters(), 'lr': 5e-5})

        optimizer = torch.optim.SGD(
            params=param_groups,
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        # save dictionary
        save_dir = fold_save_dir / f'{args.exp_name}'

        if save_dir and is_main_process():
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)

        to_restore = {"epoch": 0, "best_acc": 0.}

        kwarg = dict(
            run_variables=to_restore,
            linear_classifier=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if args.fine_tune:
            kwarg['model'] = model
        restart_from_checkpoint(
            ckp_path=save_dir / "last.pth",
            **kwarg,
        )
        start_epoch = to_restore["epoch"]
        best_acc = to_restore["best_acc"]

        best_f1 = 0
        best_result = None

        # last, best, log file
        last_w, best_w = (save_dir / "last.pth", save_dir / "best.pth")
        txt_file = save_dir / "log.txt"

        for epoch in range(start_epoch, args.epochs + 1):
            if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(
                args=args,
                model=model,
                linear_classifier=linear_classifier,
                optimizer=optimizer,
                loader=train_loader,
                epoch=epoch,
                n=args.n_last_blocks,
                depths=model.depths
            )

            scheduler.step()

            log_stats = merge_dict_with_prefix({}, train_stats, "train_")

            test_stats = validate_network(
                val_loader=val_loader,
                model=model,
                linear_classifier=linear_classifier,
                n=args.n_last_blocks,
                depths=model.depths)

            exclude = ("save_dict",)
            log_stats = merge_dict_with_prefix(log_stats, test_stats, "test_", exclude=exclude)
            log_stats['epoch'] = epoch

            LOGGER.info(f"Accuracy at epoch {epoch} test images: {test_stats['acc1']:.1f}%")

            if is_main_process():
                with txt_file.open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if args.wandb:
                    wandb.log(log_stats)

            save_dict = dict(
                epoch=epoch + 1,
                linear_classifier=de_parallel(linear_classifier).state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                best_acc=best_acc,
                best_f1=best_f1,
            )
            if args.fine_tune:
                save_dict["model"] = de_parallel(model).state_dict()

            save_on_master(save_dict, last_w)

            if is_main_process():
                # get label
                label = test_stats['save_dict']['label']
                predict = test_stats['save_dict']['predict']
                f1 = f1_score(label, predict, average='weighted')
                if f1 > best_f1:
                    # get classes name
                    name = train_loader.dataset.classes

                    best_acc, best_f1 = test_stats["acc1"], f1
                    LOGGER.info(f'Max accuracy so far: {best_acc:.4f}% F1-Score: {f1:.4f}')

                    save_on_master(save_dict, best_w)

                    cls_report = classification_report(
                        label,
                        predict,
                        labels=np.arange(0, len(name)),
                        target_names=name,
                        output_dict=True)

                    # save to csv
                    pd_cls_report = pd.DataFrame(cls_report)
                    pd_cls_report.to_csv(save_dir / 'best.csv')  # save to csv
                    best_result = cls_report

                    cm = confusion_matrix(label, predict)
                    plot_confusion_matrix(cm, name, save_dir)
                    if args.split == "patient":
                        case_analysis(test_stats['save_dict'], save_dir)

                    if args.wandb:
                        wandb.sklearn.plot_confusion_matrix(label, predict, name)
                        wandb.run.summary["best_accuracy"] = cls_report["accuracy"]
                        wandb.run.summary["best_f1"] = cls_report["weighted avg"]["f1-score"]
                        wandb.run.summary["sensitivity"] = cls_report["Malignant"]["recall"]
                        wandb.run.summary["specificity"] = cls_report["Benign"]["recall"]

            del save_dict

        # Plot acc

        LOGGER.info("Training of the supervised linear classifier on frozen features completed.\n"
                    "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
        if best_result:
            best_results.append(best_result)

        if is_main_process():
            if args.wandb:
                wandb.finish()
            # plot_txt(txt_file, keyword="acc", save_dir=save_dir, name="result")

    if is_main_process():
        reports = average_classification_reports(best_results)
        yaml_save(args.save_dir / f"report_{args.exp_name}.yaml", reports)

    torch.cuda.empty_cache()


def train(args, model, linear_classifier, optimizer, loader, epoch, n, depths):
    linear_classifier.train()
    model.train() if args.fine_tune else model.eval()

    device = get_model_device(model)  # get device

    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for batch in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = batch['img'].to(device, non_blocking=True)
        target = batch['label'].to(device, non_blocking=True)

        # forward
        if model.training is True:
            output = model.forward_return_n_last_blocks(inp, n, depths)
        else:
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


def eval_linear(cfg):
    args = get_cfg(cfg)
    main(args)


if __name__ == "__main__":
    pass
