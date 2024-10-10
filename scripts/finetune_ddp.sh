#!/bin/bash

# Set environment variables if necessary
export MASTER_PORT=23456
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# OPTION dataset=('coco_retrieval' 'flickr30k' 'coco_caption' 'snli_ve')
# OPTION criterion=('BiContrastiveLoss' 'Cosine')

# Command to run the training script with torchrun
torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} finetune_ddp.py \
    --lr 1e-4 \
    --epoch=20 \
    --batch_size=50 \
    --dataset='coco_retrieval' \
    --criterion='BiContrastiveLoss' \
    --checkpoint='checkpoints/pre-trained.pt' \
    --data_dir='datasets/coco/mscoco/' \
