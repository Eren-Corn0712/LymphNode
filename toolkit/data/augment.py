import random
import math
from typing import Tuple

import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image

import torch
from torch import Tensor
from torchvision.transforms import functional as F


class ResizePadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(self.size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)

        new_im = Image.new("RGB", (self.size, self.size))
        new_im.paste(im, ((self.size - new_size[0]) // 2,
                          (self.size - new_size[1]) // 2))
        return new_im


class Posterize(object):
    """
    Apply Posterize to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() <= self.p:
            return ImageOps.posterize(img, random.randint(4, 8))
        return img


class Equalize(object):
    """
    Apply autocontrast to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() <= self.p:
            return ImageOps.equalize(img)
        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationLymphNode(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        normalize = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0., hue=0.)],
                p=0.8
            )
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])

        # transformation for the local small crops
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        self.local_transfo = []

        for idx, l_size in enumerate(local_crops_size):
            self.local_transfo.append(transforms.Compose([
                transforms.RandomResizedCrop(l_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]))

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                crops.append(self.local_transfo[i](image))

        return crops


class DataAugmentationLymphNodeOverlapping(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        normalize = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0., hue=0.)],
                p=0.8
            )
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])

        # transformation for the local small crops
        if not isinstance(local_crops_size, tuple) or not isinstance(local_crops_size, list):
            local_crops_size = list(local_crops_size)

        if not isinstance(local_crops_number, tuple) or not isinstance(local_crops_number, list):
            local_crops_number = list(local_crops_number)

        self.local_crops_number = local_crops_number

        self.local_transfo = []

        for idx, l_size in enumerate(local_crops_size):
            self.local_transfo.append(transforms.Compose([
                transforms.RandomResizedCrop(l_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ]))

    def __call__(self, image):
        global_image1 = self.global_transfo1(image)
        global_image2 = self.global_transfo2(transforms.ToPILImage()(global_image1))
        crops = [global_image1, global_image2]
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                crops.append(self.local_transfo[i](transforms.ToPILImage()(global_image1)))

        return crops


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


def get_transform(args, name):
    if name == "lymph_node_aug":
        transform = DataAugmentationLymphNode(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.local_crops_size)
    elif name == "lymph_node_aug_overlapping":
        transform = DataAugmentationLymphNodeOverlapping(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.local_crops_size
        )

    elif name == "eval_train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    elif name == "eval_test":
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Not support for {name}")

    return transform
