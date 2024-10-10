#!/bin/bash

# Set environment variables if necessary
export CUDA_VISIBLE_DEVICES=2

# Define base paths through environment variables
export DATASET_BASE_PATH="your_datasets"
export PROJECT_PATH="your_project_path"

method="anyattack"

datasets=("coco_retrieval" "flickr30k" "snli_ve" "coco_caption")

decoders=("${PROJECT_PATH}/checkpoints/coco_bi.pt"
          "${PROJECT_PATH}/checkpoints/flickr30k_bi.pt"
          "${PROJECT_PATH}/checkpoints/snli_ve_cos.pt"
          "${PROJECT_PATH}/checkpoints/coco_cos.pt"
)

captions=("${PROJECT_PATH}/json/coco_retrieval_target.json"
          "${PROJECT_PATH}/json/flickr30k_target.json"
          "${PROJECT_PATH}/json/snli_ve_target.json"
          "${PROJECT_PATH}/json/coco_caption_target.json"
)

paths=("${DATASET_BASE_PATH}/mscoco"
       "${DATASET_BASE_PATH}/flickr30k/images"
       "${DATASET_BASE_PATH}/flickr30k/images/flickr30k-images"
       "${DATASET_BASE_PATH}/mscoco"
)

for i in {0..3}; do
    python generate_adv_img.py \
        --model_name "ViT-B/32" \
        --decoder_path "${decoders[$i]}" \
        --clean_image_path "${DATASET_BASE_PATH}/ILSVRC2012/val/" \
        --target_caption "${captions[$i]}" \
        --target_image_path "${paths[$i]}" \
        --batch_size 250 \
        --device "cuda:0" \
        --output_path "outputs" \
        --adv_imgs "${method}" \
        --dataset "${datasets[$i]}"
done
