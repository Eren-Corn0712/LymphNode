# Modified by Chunyuan Li (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import torch

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import models.vision_transformer as vits
from models import build_model

from config import config
from config import update_config
from config import save_config

from toolkit.data.augmentations import ResizePadding
from toolkit.utils.dist_utils import get_world_size
from toolkit.utils import bool_flag
from toolkit.utils.torch_utils import load_pretrained_weights, restart_from_checkpoint
from toolkit.data.lymph_dataset import KFoldLymphDataset

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


class TumorDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.ground_truth = {'B': 0, 'M': 1}
        self.benign_mass = os.path.join(self.root, 'Benign')
        self.malignant_tumor = os.path.join(self.root, 'Malignant')
        self.img_file_list = list(os.listdir(self.benign_mass)) + list(os.listdir(self.malignant_tumor))
        self.transform = transform

    def __len__(self):
        return len(self.img_file_list)

    def __getitem__(self, item):
        img_file_name = self.img_file_list[item]
        b_or_m = img_file_name.split('_')[0]
        patient_number = img_file_name.split('_')[1]
        if b_or_m == 'B':
            img_file_path = os.path.join(self.benign_mass, img_file_name)
        elif b_or_m == 'M':
            img_file_path = os.path.join(self.malignant_tumor, img_file_name)
        else:
            raise ValueError("Not a valid image")
        img = Image.open(img_file_path)

        if self.transform:
            img = self.transform(img)

        label = self.ground_truth[b_or_m]
        return img, label, patient_number


def eval_linear(args):
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        ResizePadding(size=224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.RandomVerticalFlip(),
        pth_transforms.ToTensor(),
    ])
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(size=(224, 224)),
        pth_transforms.Grayscale(num_output_channels=3),
        pth_transforms.ToTensor(),
    ])

    dataset_train = KFoldLymphDataset(root=args.data_path, transform=val_transform)
    dataset_val = KFoldLymphDataset(root=args.data_path, transform=val_transform)

    # TODO:Weight Sampler
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=None,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # plot_augmentation(val_loader)
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ create folder............. ============
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ============ building network ... ============
    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch:
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        swin_spec = config.MODEL.SPEC
        embed_dim = swin_spec['DIM_EMBED']
        depths = swin_spec['DEPTHS']
        num_heads = swin_spec['NUM_HEADS']

        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)


    # if the network is a 4-stage vision transformer (i.e. longformer)
    elif 'vil' in args.arch:
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        msvit_spec = config.MODEL.SPEC
        arch = msvit_spec.MSVIT.ARCH

        layer_cfgs = model.layer_cfgs
        num_stages = len(model.layer_cfgs)
        depths = [cfg['n'] for cfg in model.layer_cfgs]
        dims = [cfg['d'] for cfg in model.layer_cfgs]
        out_planes = model.layer_cfgs[-1]['d']
        Nglos = [cfg['g'] for cfg in model.layer_cfgs]

        print(dims)

        num_features = []
        for i, d in enumerate(depths):
            num_features += [dims[i]] * d

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)


    # if the network is a 4-stage vision transformer (i.e. CvT)
    elif 'cvt' in args.arch:
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        cvt_spec = config.MODEL.SPEC
        embed_dim = cvt_spec['DIM_EMBED']
        depths = cvt_spec['DEPTH']
        num_heads = cvt_spec['NUM_HEADS']

        print(f'embed_dim {embed_dim} depths {depths}')
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim[i])] * int(d)

        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)


    # if the network is a vanilla vision transformer (i.e. deit_tiny, deit_small, vit_base)
    elif args.arch in vits.__dict__.keys():
        depths = []
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        linear_classifier = LinearClassifier(model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)),
                                             args.num_labels)

    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    state_dict = torch.load(args.linear_weights, map_location="cpu")
    state_dict = state_dict['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    linear_classifier.load_state_dict(state_dict, strict=True)
    linear_classifier = linear_classifier.cuda()
    # linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    patient_state = dict()

    model.eval()
    linear_classifier.eval()

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            image = batch['img'].cuda(non_blocking=True)
            label = batch['label'].cuda(non_blocking=True)
            # compute output
            output = model.forward_return_n_last_blocks(image, args.n_last_blocks, args.avgpool_patchtokens, depths)
            output = linear_classifier(output)

            preds = torch.argmax(output, dim=-1)

            acc_idx = (preds == label).tolist()

            for i in range(len(acc_idx)):
                flag = acc_idx[i]
                id = batch['id'][i]
                if flag:
                    if id not in patient_state:
                        patient_state[id] = [1, 0]
                    else:
                        patient_state[id][0] += 1
                else:
                    if id not in patient_state:
                        patient_state[id] = [0, 1]
                    else:
                        patient_state[id][1] += 1

    correct = 0
    error = 0
    for k, v in patient_state.items():
        correct += v[0]
        error += v[1]
        v.insert(0, v[1] + v[0])
        acc = v[1] / v[0]
        v.append(acc)

    patient_state = pd.DataFrame.from_dict(patient_state, orient='index',
                                           columns=['Total', 'Correct', 'Error', 'Acc'])
    patient_state.to_csv(Path(args.output_dir) / 'test.csv')





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="swin_custom_patch4_window14_224_tumor.yaml",
                        type=str)

    parser.add_argument('--arch', default='swin_tiny', type=str,
                        choices=['cvt_tiny', 'swin_tiny', 'swin_small', 'swin_base', 'swin_large', 'swin', 'vil',
                                 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")

    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for DeiT-Small and to True with ViT-Base.""")

    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='runs/s_c_b_8_2048/checkpoint.pth', type=str,
                        help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='dataset', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="exp", help='Path to save logs and checkpoints')

    parser.add_argument('--num_labels', default=2, type=int, help='number of classes in a dataset')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # Load LinearClassifier Weight
    parser.add_argument('--linear_weights', default='runs/s_c_b_8_2048/teacher/11_92.68.pth', type=str, help=" ")
    parser.add_argument('--rank', default=0, type=int)
    args = parser.parse_args()
    eval_linear(args)
