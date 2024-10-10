import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TORCH_HOME'] = '/home/.cache/'

DATASET = "your_datasets"
PROJECT_PATH = "your_project_path"
datasets = ["snli_ve"]
methods = ["anyattack"]

for m in methods:
    for i in range(len(datasets)):
        print("---------------------------------------------------")
        print(f"method: {m}, Dataset: {datasets[i]}, Image Path: {datasets[i]}/{m} ")
        command = (
            f"python ve.py "
            f"--cache_path {DATASET} "
            f"--cfg_path lavis_tool/blip2/ve_snlive_vicuna7b_instruct_eval.yaml "
            f"--image_path {PROJECT_PATH}/outputs/snli_ve/{m} "
            f"--json_path {PROJECT_PATH}/json/snli_ve_adv.json "
        )
        os.system(command)