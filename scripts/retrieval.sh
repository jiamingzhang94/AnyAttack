#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/home/.cache/
# Define base paths through environment variables
export PROJECT_PATH="your_project_path"

DATA= "your_datasets"
backbones=("vitb16" "vitl14" "vitl14x336")

#datasets=("coco_retrieval" "flickr30k")
dataset="coco_retrieval"


methods=("anyattack")

for m in "${methods[@]}"; do
  for b in "${backbones[@]}"; do
    python retrieval.py \
        --cache_path ${DATA} \
        --cfg_path lavis_tool/clip/ret_coco_retrieval_eval_"${b}".yaml \
        --image_path "${PROJECT_PATH}/outputs/${dataset}/${m}" \
        --json_path "${PROJECT_PATH}/json/${dataset}_adv.json" \

  done
done

