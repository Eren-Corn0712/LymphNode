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
from toolkit.utils.torch_utils import de_parallel, time_sync
from toolkit.utils.dist_utils import save_on_master, is_main_process, get_world_size
from copy import deepcopy


def run(
        train_loader,
        val_loader,
        model,
        args,
        output_dir,
        epochs=100,
        lr=0.001,
):
    if args.arch in swin_transformer.__dict__.keys():
        embed_dim = model.embed_dim
        depths = model.depths

        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        print(f'num_features_linear {num_features_linear}')
        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)

    elif args.arch in resnet.__dict__.keys():
        depths = model.layers
        embed_dim = model.conv1.out_channels * model.layer1[-1].expansion
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d
        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        print(f'num_features_linear {num_features_linear}')
        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)
    else:
        raise ValueError(f"We not implemented {args.arch}")

    model.cuda()
    model.eval()

    linear_classifier = linear_classifier.cuda()

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr * (args.batch_size_per_gpu * 2 * get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    if output_dir and dist.get_rank() == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            # Optionally resume from a checkpoint
    best_acc = 0
    best_f1 = 0
    last_w = output_dir / "last.pth"
    best_w = output_dir / "best.pth"
    for epoch in range(0, epochs):
        # train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks,
                            args.avgpool_patchtokens, depths)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        test_stats, result = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                              args.avgpool_patchtokens, depths)
        print(f"Accuracy at epoch {epoch} test images: {test_stats['acc1']:.1f}%")
        log_stats = {**{k: v for k, v in log_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()}}
        if is_main_process():
            with (Path(output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": (de_parallel(linear_classifier)).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        save_on_master(save_dict, str(last_w))

        f1 = f1_score(result['target'], result['predict'], average='weighted')
        if is_main_process():
            if f1 > best_f1:
                name = ['Benign', 'Malignant']
                best_acc, best_f1 = test_stats["acc1"], f1
                print(f'Max accuracy so far: {best_acc:.2f}% F1-Score: {f1:.2f}')

                save_on_master(save_dict, str(best_w))
                pd.DataFrame(classification_report(result['target'], result['predict'],
                                                   target_names=name, output_dict=True)).to_csv(output_dir / 'best.csv')
                cm = confusion_matrix(result['target'], result['predict'])
                cm = pd.DataFrame(cm, name, name)
                plt.figure(figsize=(9, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap='BuGn')
                plt.xlabel("prediction")
                plt.ylabel("label (ground truth)")
                plt.savefig(output_dir / "best_confusion_matrix.png")

    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, depths):
    linear_classifier.train()
    model.eval()

    metric_logger = toolkit.utils.logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', toolkit.utils.logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for batch in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = batch['img']
        target = batch['label']
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.forward_return_n_last_blocks(inp, n, depths)

        # print(f'output {output.shape}')
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, _ = toolkit.utils.torch_utils.accuracy(output, target, topk=(1, 2))

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        time_sync()
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, depths):
    linear_classifier.eval()
    model.eval()
    metric_logger = toolkit.utils.logger.MetricLogger(delimiter="  ")
    header = 'Test:'
    result = {'target': [],
              'predict': []}
    for batch in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = batch['img']
        target = batch['label']

        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model.forward_return_n_last_blocks(inp, n, depths)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, _ = toolkit.utils.torch_utils.accuracy(output, target, topk=(1, 2))

        _, predict = torch.max(output.data, 1)

        result['target'].extend(target.view(-1).detach().cpu().numpy())
        result['predict'].extend(predict.view(-1).detach().cpu().numpy())

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, result


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)
