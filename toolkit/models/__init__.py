import toolkit.models.swin_transformer as swin_transformer
import toolkit.models.resnet as resnet

from toolkit.models.head import DINOHead, MixDINOHead, LinearClassifier, TransformerHead, MultiLevelHead

from toolkit.models.care import AttnHead
from toolkit.utils import LOGGER

import torchvision


def create_linear_layer(model, args):
    if args.arch in swin_transformer.__dict__.keys():
        embed_dim = model.embed_dim
        depths = model.depths

        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        LOGGER.info(f"num_features: {num_features}")
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        LOGGER.info(f'num_features_linear {num_features_linear}')
        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)

    elif args.arch in resnet.__dict__.keys():
        depths = model.layers
        model.depths = model.layers
        embed_dim = model.conv1.out_channels * model.layer1[-1].expansion
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d

        LOGGER.info(f"num_features: {num_features}")
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        LOGGER.info(f'num_features_linear {num_features_linear}')
        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)

    else:
        raise ValueError(f"We not implemented {args.arch}")

    return linear_classifier


def create_model(args):
    if args.arch in torchvision.models.__dict__.keys():
        model = torchvision.models.__dict__[args.arch]()
    elif args.arch in torchvision.models.__dict__.keys():
        model = torchvision.models.__dict__[args.arch]()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    LOGGER.info(f"Model is built {args.arch} network.")
    return model


def create_teacher_student(args):
    #
    if args.arch in swin_transformer.__dict__.keys():
        student = swin_transformer.__dict__[args.arch]()
        teacher = swin_transformer.__dict__[args.arch]()

    elif args.arch in resnet.__dict__.keys():
        student = resnet.__dict__[args.arch]()
        teacher = resnet.__dict__[args.arch]()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    for model in [student, teacher]:
        if hasattr(args, "use_head_prediction") and args.use_head_prediction:
            model.head = DINOHead(
                model.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
        if hasattr(args, "use_dense_prediction") and args.use_dense_prediction:
            setattr(model, "use_dense_prediction", args.use_dense_prediction)
            model.dense_head = DINOHead(
                model.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer
            )
        if hasattr(args, "use_mix_prediction") and args.use_mix_prediction:
            setattr(model, "use_mix_prediction", args.use_mix_prediction)
            model.mix_head = MixDINOHead(
                model.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer
            )
        if hasattr(args, "use_trans_prediction") and args.use_trans_prediction:
            setattr(model, "use_trans_prediction", args.use_trans_prediction)
            model.trans_head = TransformerHead(
                model.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
        if hasattr(args, "use_multi_level") and args.use_multi_level:
            setattr(model, "use_multi_level", args.use_multi_level)
            # only support for resnet18
            model.multi_head = MultiLevelHead(
                [model.layer1[-1].conv2.out_channels,
                 model.layer2[-1].conv2.out_channels,
                 model.layer3[-1].conv2.out_channels,
                 model.layer4[-1].conv2.out_channels],
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer
            )

    LOGGER.info(f"Student and Teacher are built: they are both {args.arch} network.")
    return teacher, student
