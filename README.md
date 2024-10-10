
# AnyAttack: Official Code for "[AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models](https://arxiv.org/abs/2410.05346)"

This repository provides the official implementation of the paper "AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models." The code includes setup instructions, data preparation, model training, and evaluation scripts for replicating the results presented in the paper.

## Installation

### Step 1: Environment Setup

1. **Create separate Conda environments for LAVIS and Mini-GPT4**:  
   Since BLIP, BLIP2, and InstructBLIP rely on the [LAVIS library](https://github.com/salesforce/LAVIS), while Mini-GPT4 uses a different set of packages that may conflict with LAVIS dependencies, it's recommended to use two separate Conda environments.

   - **LAVIS environment**: Follow the instructions [here](https://github.com/salesforce/LAVIS) to set up.
   - **Mini-GPT4 environment**: Set up according to [Mini-GPT4's installation guide](https://github.com/Vision-CAIR/MiniGPT-4).

2. **Data Preparation**:
   - **Datasets**:
     - MSCOCO and Flickr30K datasets can be found [here](https://opensource.salesforce.com/LAVIS//latest/benchmark).
     - **LAION-400M**: Use the [img2dataset tool](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md) to download the LAION-400M dataset for pretraining.
     - **ImageNet**: Download and prepare the ImageNet dataset separately.

### Step 2: Download Checkpoints and JSON Files

- Pretrained models and configuration files can be downloaded from [OneDrive](https://gohkust-my.sharepoint.com/:u:/g/personal/jmzhang_ust_hk/EdoO5KyVBH1FhPVr1kSYWh0B61oR9MYN9_EYmrCFBKnLsQ?e=IfkDmh). Place the downloaded files in the current project directory.

### Step 3 (Optional): Training and Fine-tuning

You can either use the pretrained weights from Step 2 or train the models from scratch.

1. **Pretraining on LAION-400M**:
   - Run the pretraining script:
     ```bash
     ./scripts/main.sh
     ```
   - Replace `"YOUR_LAION_DATASET"` with the path to your LAION-400M dataset.

2. **Fine-tuning on downstream datasets**:
   - Use the script:
     ```bash
     ./scripts/finetune_ddp.sh
     ```
   - Edit the script to set the desired `dataset` and `criterion` parameters. Make sure to adjust the `data_dir` parameter based on the chosen dataset.

### Step 4: Generate Adversarial Images

Use the pretrained decoder to generate adversarial images:

1. Run the script:
   ```bash
   ./scripts/generate_adv_img.sh
   ```
2. Store the datasets from Step 1 under the `DATASET_BASE_PATH` directory, and set `PROJECT_PATH` to the current project directory.

### Step 5: Evaluation

Evaluate the trained models on different tasks:

1. **Image-text retrieval**: 
   ```bash
   ./scripts/retrieval.sh
   ```
2. **Multimodal classification**:
   ```bash
   python ./scripts/classification_shell.py
   ```
3. **Image captioning**:
   ```bash
   python ./scripts/caption_shell.py
   ```

For each task, options and parameters are provided in the scripts as comments.

## Citation

If you find this work useful, please cite:
```bibtex
@article{zhang2024anyattack,
      title={AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models}, 
      author={Jiaming Zhang and Junhong Ye and Xingjun Ma and Yige Li and Yunfan Yang and Jitao Sang and Dit-Yan Yeung},
      year={2024},
      journal={arXiv preprint arXiv:2410.05346},
}
```
