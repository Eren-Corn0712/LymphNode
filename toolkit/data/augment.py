import random
import torchvision.transforms as transforms
from PIL import ImageFilter, ImageOps, Image


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
            (0.95, 1.00),
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
