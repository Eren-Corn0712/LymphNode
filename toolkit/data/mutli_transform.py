from abc import ABCMeta

import random
import numpy as np
import albumentations as A
import albumentations.pytorch as A_P
from PIL import ImageFilter, ImageOps, Image
from toolkit.utils import LOGGER, colorstr
from torchvision.transforms import (transforms, InterpolationMode)


class IdentityAlbumentations(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return kwargs


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


class ColorJitter(transforms.RandomApply):
    """
    Apply ColorJitter to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, brightness=0, contrast=0, saturation=0, hue=0):
        transform = transforms.ColorJitter(brightness=brightness,
                                           contrast=contrast,
                                           saturation=saturation,
                                           hue=hue)
        super().__init__(transforms=[transform], p=p)


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


class AutoContrast(object):
    """
    Apply autocontrast to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.autocontrast(img, cutoff=10)
        else:
            return img


class Equalize(object):
    """
    Apply Equalize to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.equalize(img)
        else:
            return img


class MultiDataAugmentation(object):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        # It will initial on children class
        self.geometric_augmentation_global = None
        self.geometric_augmentation_local = None
        self.global_trans1 = None
        self.global_trans2 = None
        self.local_trans = None

    def __call__(self, image):
        """
        If necessary, you can overwrite this function.
        :param image: PIL Image
        :return: List of Augmented Image
        """
        output = []
        geo_image1, geo_image2 = (
            self.geometric_augmentation_global(image) for _ in range(2)
        )
        global_1, global_2 = self.global_trans1(geo_image1), self.global_trans2(geo_image2)

        output.extend((global_1, global_2))

        output.extend(
            (
                self.local_trans(
                    self.geometric_augmentation_local(
                        image
                    )
                ) for _ in range(self.local_crops_number))
        )
        return output

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}," \
            f"global_crops_scale:{self.global_crops_scale}" \
            f"local_crops_scale: {self.local_crops_scale}" \
            f"local_crops_number: {self.local_crops_number}" \
            f"global_crops_size: {self.global_crops_size}" \
            f"local_crops_size: {self.local_crops_size}"
        return s


class DataAugmentationLymphNode(MultiDataAugmentation, metaclass=ABCMeta):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        super().__init__(
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size,
            local_crops_size,
        )

        # Data Augmentation set
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    ratio=(1.0, 1.0),  # make sure it's square
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    ratio=(1.0, 1.0),  # make sure it's square
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        color_jitter = transforms.Compose(
            [
                ColorJitter(p=0.8, brightness=0.5, contrast=0.75, saturation=0.0, hue=0.0)
            ]
        )

        global_trans1_extra = transforms.Compose(
            [
                GaussianBlur(p=1.0),
            ]
        )
        global_trans2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )
        local_trans_extra = GaussianBlur(p=0.5)

        normalize = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.global_trans1 = transforms.Compose([color_jitter, global_trans1_extra, normalize])
        self.global_trans2 = transforms.Compose([color_jitter, global_trans2_extra, normalize])
        self.local_trans = transforms.Compose([color_jitter, local_trans_extra, normalize])


class DataAugmentationLymphNodeOld(MultiDataAugmentation, metaclass=ABCMeta):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        super().__init__(
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size,
            local_crops_size,
        )

        # Data Augmentation set
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.Resize((global_crops_size, global_crops_size),
                                  InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    ratio=(1.0, 1.0),  # make sure it's square
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        global_trans1_extra = transforms.Compose(
            [
                Identity()
            ]
        )
        global_trans2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                Solarization(0.2),
                Equalize(0.1),
                AutoContrast(0.1),
            ]
        )
        local_trans_extra = GaussianBlur(p=0.5)

        normalize = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

        self.global_trans1 = transforms.Compose([global_trans1_extra, normalize])
        self.global_trans2 = transforms.Compose([global_trans2_extra, normalize])
        self.local_trans = transforms.Compose([local_trans_extra, normalize])


class MultiDataAugAlbumentations(object):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        LOGGER.info(f"{self.__class__.__name__}")
        LOGGER.info(f"global_crops_scale: {global_crops_scale}")
        LOGGER.info(f"local_crops_scale: {local_crops_scale}")
        LOGGER.info(f"global_crops_size: {global_crops_size}")
        LOGGER.info(f"local_crops_size: {local_crops_size}")
        LOGGER.info(f"local_crops_number: {local_crops_number}")

        # It will initial on children class
        self.geometric_augmentation_global = None
        self.geometric_augmentation_local = None
        self.global_trans1 = None
        self.global_trans2 = None
        self.local_trans = None

    def __call__(self, image):
        image = np.asarray(image)  # pil to numpy
        output = []  # save all the output
        geo_image1, geo_image2 = (
            self.geometric_augmentation_global(image=image) for _ in range(2)
        )
        global_1 = self.global_trans1(image=geo_image1['image'])['image']
        global_2 = self.global_trans2(image=geo_image2['image'])['image']

        output.extend((global_1, global_2))

        output.extend(
            (
                self.local_trans(
                    image=self.geometric_augmentation_local(
                        image=image
                    )['image']
                )['image'] for _ in range(self.local_crops_number)
            )
        )
        output = [x.float() / 255.0 for x in output]
        return [x.repeat(3, 1, 1) if x.shape[0] == 1 else x for x in output]


class AlbumentationsLymphNode(MultiDataAugAlbumentations, metaclass=ABCMeta):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        super().__init__(global_crops_scale, local_crops_scale, local_crops_number, global_crops_size, local_crops_size)
        self.geometric_augmentation_global = A.Compose(
            [
                A.Resize(
                    height=self.local_crops_size,
                    width=self.local_crops_size,
                    interpolation=3,
                    p=1.0
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ]
        )

        self.geometric_augmentation_local = A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.local_crops_size,
                    width=self.local_crops_size,
                    scale=self.local_crops_scale,
                    ratio=(1.0, 1.0),
                    interpolation=3,
                    p=1.0
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ]
        )
        global_trans1_extra = IdentityAlbumentations()
        global_trans2_extra = IdentityAlbumentations()
        local_trans_extra = IdentityAlbumentations()
        normalize = A.Compose(
            [
                A_P.ToTensorV2(),
            ]
        )
        self.global_trans1 = A.Compose([global_trans1_extra, normalize])
        self.global_trans2 = A.Compose([global_trans2_extra, normalize])
        self.local_trans = A.Compose([local_trans_extra, normalize])


class AlbumentationsLymphNodeV1(MultiDataAugAlbumentations, metaclass=ABCMeta):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        super().__init__(global_crops_scale, local_crops_scale, local_crops_number, global_crops_size, local_crops_size)

        self.geometric_augmentation_global = A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.global_crops_size,
                    width=self.global_crops_size,
                    scale=self.global_crops_scale,
                    ratio=(1.0, 1.0),
                    interpolation=3,
                    p=1.0
                ),
            ]
        )

        self.geometric_augmentation_local = A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.local_crops_size,
                    width=self.local_crops_size,
                    scale=self.local_crops_scale,
                    ratio=(1.0, 1.0),
                    interpolation=3,
                    p=1.0
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ]
        )

        color_jitter = A.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0, p=0.8)

        global_trans1_extra = IdentityAlbumentations()
        global_trans2_extra = IdentityAlbumentations()
        local_trans_extra = IdentityAlbumentations()

        normalize = A.Compose(
            [
                A_P.ToTensorV2(),
            ]
        )
        self.global_trans1 = A.Compose([color_jitter, global_trans1_extra, normalize])
        self.global_trans2 = A.Compose([color_jitter, global_trans2_extra, normalize])
        self.local_trans = A.Compose([color_jitter, local_trans_extra, normalize])




class AlbumentationsLymphNodeV2(MultiDataAugAlbumentations, metaclass=ABCMeta):
    def __init__(
            self,
            global_crops_scale,
            local_crops_scale,
            local_crops_number,
            global_crops_size=224,
            local_crops_size=96,
    ):
        super().__init__(global_crops_scale, local_crops_scale, local_crops_number, global_crops_size, local_crops_size)

        self.geometric_augmentation_global = A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.global_crops_size,
                    width=self.global_crops_size,
                    scale=self.global_crops_scale,
                    ratio=(1.0, 1.0),
                    interpolation=3,
                    p=1.0
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ]
        )

        self.geometric_augmentation_local = A.Compose(
            [
                A.RandomResizedCrop(
                    height=self.local_crops_size,
                    width=self.local_crops_size,
                    scale=self.local_crops_scale,
                    ratio=(1.0, 1.0),
                    interpolation=3,
                    p=1.0
                ),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ]
        )

        color_jitter = A.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0, p=0.8)

        global_trans1_extra = IdentityAlbumentations()
        global_trans2_extra = IdentityAlbumentations()
        local_trans_extra = IdentityAlbumentations()

        normalize = A.Compose(
            [
                A_P.ToTensorV2(),
            ]
        )
        self.global_trans1 = A.Compose([color_jitter, global_trans1_extra, normalize])
        self.global_trans2 = A.Compose([color_jitter, global_trans2_extra, normalize])
        self.local_trans = A.Compose([color_jitter, local_trans_extra, normalize])
