#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=/home/.cache/
# Define base paths through environment variables
export PROJECT_PATH="your_project_path"
dataset="coco_caption"
method="anyattack"

python generate_response.py  \
    --cfg_path 'minigpt4_llama2_eval.yaml' \
    --prompt "Describe this image in one short sentence only." \
    --data_path "${PROJECT_PATH}/json/${dataset}_adv.json" \
    --image_path  "${PROJECT_PATH}/outputs/${dataset}/${method}" \
    --gt_path "${PROJECT_PATH}/json/coco_caption_test_gt_adv.json" \
    --output_path "${PROJECT_PATH}/outputs/minigpt4-llama2/${method}.json" \
    --llama_path 'meta-llama/Llama-2-7b-chat-hf' \
    --ckpt_path '/YOUR/CHECKPOINT/PATH'
