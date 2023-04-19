#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# shellcheck disable=SC2054
MODELS=resnet18
DIM=4096

python -m torch.distributed.run --nproc_per_node=2 --master_port 25671 main_esvit_lymph.py \
  --arch $MODELS \
  --data_path dataset leaf_tumor_video \
  --save_dir runs/2023-04-19-w-dense-w-attn/$MODELS/$DIM \
  --batch_size_per_gpu 256 \
  --momentum_teacher 0.9995 \
  --use_bn_in_head True \
  --epochs 300 \
  --freeze_last_layer 1 \
  --warmup_epochs 10 \
  --warmup_teacher_temp 0.04 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer True \
  --out_dim $DIM \
  --aug-opt lymph_node_aug \
  --global_crops_scale 0.50 1.0 \
  --local_crops_number 8 \
  --local_crops_scale 0.25 0.75 \
  --local_crops_size 96 \
  --num_workers 12 \
  --lr 5e-5 \
  --use_dense_prediction True \
  --use_attention_head True \
  --linear_epochs 150 \
  --linear_lr 0.001

python -m torch.distributed.run --nproc_per_node=2 --master_port 25671 main_esvit_lymph.py \
  --arch $MODELS \
  --data_path dataset leaf_tumor_video \
  --save_dir runs/2023-04-19-w-dense-wo-attn/$MODELS/$DIM \
  --batch_size_per_gpu 256 \
  --momentum_teacher 0.9995 \
  --use_bn_in_head True \
  --epochs 300 \
  --freeze_last_layer 1 \
  --warmup_epochs 10 \
  --warmup_teacher_temp 0.04 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer True \
  --out_dim $DIM \
  --aug-opt lymph_node_aug \
  --global_crops_scale 0.50 1.0 \
  --local_crops_number 8 \
  --local_crops_scale 0.25 0.75 \
  --local_crops_size 96 \
  --num_workers 12 \
  --lr 5e-5 \
  --use_dense_prediction True \
  --use_attention_head False \
  --linear_epochs 150 \
  --linear_lr 0.001

python -m torch.distributed.run --nproc_per_node=2 --master_port 25671 main_esvit_lymph.py \
  --arch $MODELS \
  --data_path dataset leaf_tumor_video \
  --save_dir runs/2023-04-19-wo-dense-w-attn/$MODELS/$DIM \
  --batch_size_per_gpu 256 \
  --momentum_teacher 0.9995 \
  --use_bn_in_head True \
  --epochs 300 \
  --freeze_last_layer 1 \
  --warmup_epochs 10 \
  --warmup_teacher_temp 0.04 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer True \
  --out_dim $DIM \
  --aug-opt lymph_node_aug \
  --global_crops_scale 0.50 1.0 \
  --local_crops_number 8 \
  --local_crops_scale 0.25 0.75 \
  --local_crops_size 96 \
  --num_workers 12 \
  --lr 5e-5 \
  --use_dense_prediction False \
  --use_attention_head True \
  --linear_epochs 150 \
  --linear_lr 0.001

python -m torch.distributed.run --nproc_per_node=2 --master_port 25671 main_esvit_lymph.py \
  --arch $MODELS \
  --data_path dataset leaf_tumor_video \
  --save_dir runs/2023-04-19-wo-dense-wo-attn/$MODELS/$DIM \
  --batch_size_per_gpu 256 \
  --momentum_teacher 0.9995 \
  --use_bn_in_head True \
  --epochs 300 \
  --freeze_last_layer 1 \
  --warmup_epochs 10 \
  --warmup_teacher_temp 0.04 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer True \
  --out_dim $DIM \
  --aug-opt lymph_node_aug \
  --global_crops_scale 0.50 1.0 \
  --local_crops_number 8 \
  --local_crops_scale 0.25 0.75 \
  --local_crops_size 96 \
  --num_workers 12 \
  --lr 5e-5 \
  --use_dense_prediction False \
  --use_attention_head False \
  --linear_epochs 150 \
  --linear_lr 0.001
