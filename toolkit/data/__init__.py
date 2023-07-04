from torchvision.transforms import transforms, InterpolationMode

from toolkit.data.augmentations import ClassificationPresetTrain, ClassificationPresetEval
from toolkit.data.mutli_transform import DataAugmentationLymphNode, DataAugmentationLymphNodeOld, \
    AlbumentationsLymphNode, AlbumentationsLymphNodeV1, AlbumentationsLymphNodeV2
from toolkit.utils import LOGGER


def create_transform(args, name):
    if name == "lymph_node_aug":
        transform = DataAugmentationLymphNode(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            local_crops_number=args.local_crops_number,
            global_crops_size=args.global_crops_size,
            local_crops_size=args.local_crops_size)
    elif name == "lymph_node_aug_1":
        transform = DataAugmentationLymphNodeOld(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            local_crops_number=args.local_crops_number,
            global_crops_size=args.global_crops_size,
            local_crops_size=args.local_crops_size)


    elif name == "A0":
        transform = AlbumentationsLymphNode(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            local_crops_number=args.local_crops_number,
            global_crops_size=args.global_crops_size,
            local_crops_size=args.local_crops_size
        )
    elif name == "A1":
        transform = AlbumentationsLymphNodeV1(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            local_crops_number=args.local_crops_number,
            global_crops_size=args.global_crops_size,
            local_crops_size=args.local_crops_size
        )

    elif name == "A2":
        transform = AlbumentationsLymphNodeV2(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            local_crops_number=args.local_crops_number,
            global_crops_size=args.global_crops_size,
            local_crops_size=args.local_crops_size
        )

    elif name == "eval_train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.global_crops_size,
                                         scale=args.global_crops_scale),
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


    elif name == "preset_train":
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        transform = ClassificationPresetTrain(
            crop_size=args.train_crop_size,
            interpolation=InterpolationMode(args.interpolation),
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
            ra_magnitude=ra_magnitude,
            augmix_severity=augmix_severity,
        )
    elif name == "preset_test":
        transform = ClassificationPresetEval(
            crop_size=args.val_crop_size,
            resize_size=args.val_resize_size,
            interpolation=InterpolationMode(args.interpolation)
        )
    else:
        raise ValueError(f"Not support for {name}")
    LOGGER.info(f"Data Augmentation: {transform.__class__.__name__}")
    return transform
