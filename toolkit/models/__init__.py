import toolkit.models.swin_transformer as swin_transformer
import toolkit.models.resnet as resnet

from toolkit.models.head import DINOHead
from toolkit.models.care import AttnHead


def create_teacher_student(args):
    if args.arch in swin_transformer.__dict__.keys():
        student = swin_transformer.__dict__[args.arch]()
        teacher = swin_transformer.__dict__[args.arch](is_teacher=True)

        for model in [student, teacher]:
            model.head = DINOHead(
                model.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            if args.use_dense_prediction:
                setattr(model, "use_dense_prediction", args.use_dense_prediction)
                model.dense_head = DINOHead(
                    model.num_features,
                    args.out_dim,
                    use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer)

    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in resnet.__dict__.keys():
        student = resnet.__dict__[args.arch]()
        teacher = resnet.__dict__[args.arch]()
        for model in [student, teacher]:
            model.head = DINOHead(
                model.num_features,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            )
            if args.use_dense_prediction:
                setattr(model, "use_dense_prediction", args.use_dense_prediction)
                model.dense_head = DINOHead(
                    model.num_features,
                    args.out_dim,
                    use_bn=args.use_bn_in_head,
                    norm_last_layer=args.norm_last_layer)

            if args.use_attention_head:
                setattr(model, "use_attention_head", args.use_attention_head)
                model.attn_head = AttnHead(
                    model.num_features,
                    args.out_dim,
                    norm_last_layer=args.norm_last_layer
                )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    return teacher, student
