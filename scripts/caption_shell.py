import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TORCH_HOME'] = '/home/.cache/'

DATASET = "your_datasets"
PROJECT_PATH = "your_project_path"
datasets = ["coco_caption"]
methods = ['anyattack']

model = "blip2/caption_coco_opt2.7b_eval.yaml"
# OPTION ["lavis_tool/blip2/caption_coco_opt2.7b_eval.yaml", "lavis_tool/blip/caption_coco_eval.yaml", "blip2/caption_coco_vicuna7b_instruct_eval.yaml" ]

for m in methods:
    for i in range(len(datasets)):
        print("---------------------------------------------------")
        print(f"Dataset: {datasets[i]}, Image Path: {datasets[i]}/{m} ")

        command = (
            f"python caption.py "
            f"--cache_path {DATASET} "
            f"--image_path {PROJECT_PATH}/outputs/{datasets[i]}/{m} "
            f"--json_path {PROJECT_PATH}/json/{datasets[i]}_adv.json "
            f"--gt_path {PROJECT_PATH}/json/coco_caption_test_gt_adv.json"
            )
        command += f" --cfg_path lavis_tool/blip/caption_coco_eval.yaml"
        os.system(command)