#!/bin/bash

# Set environment variables if necessary
export MASTER_PORT=23456
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

laion="YOUR_LAION_DATASET" # e.g., "big_datasets/laion-400m/laion400m-data"

# Command to run the training script with torchrun
torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} main_ddp.py \
    --tar_dir= "${laion}"\
    --lr 1e-4 \
    --epoch=5 \
    --batch_size=600 \
    --dist_url="tcp://127.0.0.1:${MASTER_PORT}" \
