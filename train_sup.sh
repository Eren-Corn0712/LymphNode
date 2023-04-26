#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# shellcheck disable=SC2054
MODELS=resnet18

python -m torch.distributed.run --nproc_per_node=1 --master_port 25671 supervise_train.py \
  --model $MODELS \
  --data-path dataset leaf_tumor_video \
  --save-dir runs/2023-04-24-sup/$MODELS \
  --batch-size 256 \
  --epochs 300 \
  --lr 0.001 \
  --sync-bn \
  --amp \
  --model-ema
