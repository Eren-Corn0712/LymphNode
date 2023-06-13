import torch
import torch.nn as nn
import types
import argparse
import cv2
import numpy as np
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import Subset
import torchvision.transforms as transforms

from toolkit.models import create_teacher_student
from toolkit.utils import bool_flag, yaml_load
from toolkit.data.lymph_dataset import KFoldLymphDataset
from toolkit.models import create_linear_layer
from toolkit.utils.torch_utils import load_pretrained_weights, load_pretrained_linear_weights
from toolkit.utils.plots import plot_cam
from toolkit.data.augmentations import create_transform
from pathlib import Path
import pathlib
from tqdm import tqdm

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def get_args_parser():
    parser = argparse.ArgumentParser("CAM", add_help=False)

    parser.add_argument('--data_path', default=['dataset_clean'], type=str,
                        nargs='+', help='Please specify path to the ImageNet training data.')
    parser.add_argument('--backbone_yaml', default="runs_esvit/20230602_DINO_1024/args.yaml", type=str)
    parser.add_argument('--downstream_yaml', default="runs_esvit/20230602_DINO_1024/args_best_linear_fine_tune.yaml",
                        type=str)
    return parser


def to_uint8(x):
    return np.uint8(x * 255)


class ModelWrapper(nn.Module):
    def __init__(self, args, model, linear):
        super().__init__()
        self.args = args
        self.model = model
        self.linear = linear

    def forward(self, x):
        x = self.model.forward_return_n_last_blocks(x, self.args.n_last_blocks, self.model.depths)
        x = self.linear(x)
        return x


def main_cam(args):
    th = 255 // 2
    # Pretrained model path
    backbone_args = yaml_load(args.backbone_yaml)
    backbone_args = types.SimpleNamespace(**backbone_args)

    downstream_yaml = yaml_load(args.downstream_yaml)
    downstream_yaml = types.SimpleNamespace(**downstream_yaml)

    # Create a model and eval mode
    model, _ = create_teacher_student(backbone_args)
    linear_classifier = create_linear_layer(model, downstream_yaml)

    cam_model = ModelWrapper(args=downstream_yaml, model=model, linear=linear_classifier)

    # The target Layer
    target_layers = [cam_model.model.layer4[-1]]

    k_fold_dataset = KFoldLymphDataset(backbone_args.data_path, n_splits=5, shuffle=True,
                                       random_state=backbone_args.seed)
    idx_to_class = {v: k[0] for k, v in k_fold_dataset.class_to_idx.items()}

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 文字顏色為白色
    thickness = 2

    for k, (train_set, test_set) in enumerate(k_fold_dataset.generate_fold_dataset()):
        fold_path = Path(backbone_args.save_dir) / f"{int(k + 1)}-fold"
        fold_model_path = (fold_path / "best").with_suffix(".pth")
        # TO DO: Modfiy this name
        fold_linear_path = (fold_path / downstream_yaml.exp_name / "best").with_suffix(".pth")

        if downstream_yaml.fine_tune:
            load_pretrained_linear_weights(cam_model.model, str(fold_linear_path), key='model')
        else:
            load_pretrained_weights(cam_model.model, str(fold_model_path), checkpoint_key="teacher")

        load_pretrained_linear_weights(cam_model.linear, str(fold_linear_path))

        if downstream_yaml.aug_opt == "eval":
            train_set.transform = create_transform(downstream_yaml, "eval_train")
            test_set.transform = create_transform(downstream_yaml, "eval_test")
        elif downstream_yaml.aug_opt == "eval_norm":
            train_set.transform = create_transform(downstream_yaml, "eval_train_norm")
            test_set.transform = create_transform(downstream_yaml, "eval_test_norm")
        else:
            raise ValueError(f"Augmentation key {downstream_yaml.aug_opt} is not support!")

        cam = LayerCAM(model=cam_model, target_layers=target_layers, use_cuda=True)
        vid_writer = {}

        cam_model.eval()
        for idx, data in enumerate(tqdm(test_set)):
            inp = data['img']
            target = data['label']
            im_file = Path(data['im_file'])

            targets = [ClassifierOutputTarget(target)]
            output = cam_model(inp[None, ...].cuda())
            grayscale_cam = cam(input_tensor=inp[None, ...], targets=targets)

            _, pred = torch.max(output.data, 1)
            pred = pred.view(-1).item()
            np_inp = inp.numpy().transpose(1, 2, 0)
            grayscale_cam = grayscale_cam.transpose(1, 2, 0)

            semantic_inp = np.zeros_like(np_inp)
            mask_index = np.repeat(to_uint8(grayscale_cam) > th, 3, axis=-1)
            semantic_inp[mask_index] = True
            semantic_inp = np.where(semantic_inp, 255, 0).astype(np.uint8)
            visualization = show_cam_on_image(np_inp,
                                              grayscale_cam,
                                              use_rgb=False)

            result = np.hstack((to_uint8(np_inp), visualization, semantic_inp))
            padding_size = 100
            result_padded = cv2.copyMakeBorder(result, padding_size, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # create and save mp4 file
            save_path = fold_path / data['type_name'] / data['patient_id']
            save_path.mkdir(parents=True, exist_ok=True)

            key = f"{data['type_name']}-{data['patient_id']}"
            fps, w, h = 5, result_padded.shape[1], result_padded.shape[0]

            if key not in vid_writer:
                save_path = Path(save_path / "LayerCAM").with_suffix('.mp4')  # force *.mp4 suffix on results videos
                vid_writer[key] = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if save_path.exists():
                    print(save_path.resolve())
            else:
                text = f"ID:{data['patient_id']}-label:{data['type_name'][0]}-pred:{idx_to_class[pred]}"
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

                text_x = (result_padded.shape[1] - text_size[0]) // 2  # 水平居中
                text_y = padding_size - text_size[1] + 10  # 文字的垂直位置

                cv2.putText(result_padded, text, (text_x, text_y), font, font_scale, font_color, thickness)

                vid_writer[key].write(result_padded)

            # if target == pred and target == 1:
            #     plot_cam(result,
            #              save_dir=fold_path,
            #              patient_id=data['patient_id'],
            #              disease_type=data['type_name'],
            #              name=im_file.name + "_correct",
            #              predict=str(idx_to_class[pred]))
            # else:
            #     plot_cam(result,
            #              save_dir=fold_path,
            #              patient_id=data['patient_id'],
            #              disease_type=data['type_name'],
            #              name=im_file.name + "_error",
            #              predict=str(idx_to_class[pred]))
        if k == 0:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser('EsViT', parents=[get_args_parser()])
    args = parser.parse_args()
    main_cam(args)
