import torchvision.transforms as transforms
from toolkit.esvit.utils import GaussianBlur


class DataAugmentationLymphNode(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, local_crops_size=96):
        normalize = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        Random_HF_VF = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, ratio=(1.0, 1.0)),
            Random_HF_VF,
            transforms.ElasticTransform(alpha=50.0),
            GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, ratio=(1.0, 1.0)),
            Random_HF_VF,
            transforms.AugMix(),
            GaussianBlur(p=0.5),
            normalize,
        ])

        self.local_crops_number = local_crops_number

        self.local_transfo = []

        for idx, l_size in enumerate(local_crops_size):
            self.local_transfo.append(transforms.Compose([
                transforms.RandomResizedCrop(l_size, scale=local_crops_scale, ratio=(1.0, 1.0)),
                Random_HF_VF,
                GaussianBlur(p=0.5),
                normalize,
            ]))

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for i, n_crop in enumerate(self.local_crops_number):
            for _ in range(n_crop):
                crops.append(self.local_transfo[i](image))
        return crops


class FineTuneAugmentation(object):
    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        self.test_transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
        ])

    def __call__(self, image):
        return self.train_transform(image) if self.is_train else self.test_transform(image)
