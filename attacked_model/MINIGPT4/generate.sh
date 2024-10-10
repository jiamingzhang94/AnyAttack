#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export TORCH_HOME=/home/dycpu6_8tssd1/jmzhang/.cache/

methods=("en_Cosine")
# losses = ["Cosine"]

#     --image_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/outputs/coco_caption/early_Cosine' \
#    --data_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/outputs/coco_caption_early_Cosine.json' \
# --image_path '/home/dycpu6_8tssd1/jmzhang/datasets/mscoco' \
# --data_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/json/coco_karpathy_test.json' \


for m in "${methods[@]}"; do
  python generate_response.py  \
      --cfg_path 'minigpt4v2_llama2_eval.yaml' \
      --prompt "Describe this image in one short sentence only." \
      --data_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/outputs/coco_caption_early_Cosine.json' \
      --image_path "/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/outputs/coco_caption/${m}" \
      --gt_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/json/coco_caption_test_gt_adv.json' \
      --output_path "output/minigpt4V2-llama2/${m}.json" \
      --llama_path 'meta-llama/Llama-2-7b-chat-hf' \
      --ckpt_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/checkpoints/minigptv2_checkpoint.pth'
done

#for m in "${methods[@]}"; do
#  python generate_response.py  \
#    --cfg_path 'minigpt4_llama2_eval.yaml' \
#    --prompt "Describe this image in one short sentence only." \
#    --data_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/outputs/coco_caption_early_Cosine.json' \
#    --image_path "/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/outputs/coco_caption/${m}" \
#    --gt_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/json/coco_caption_test_gt_adv.json' \
#    --output_path "output/minigpt4-llama2/${m}.json" \
#    --llama_path 'meta-llama/Llama-2-7b-chat-hf' \
#    --ckpt_path '/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/checkpoints/pretrained_minigpt4_llama2_7b.pth'
#done
