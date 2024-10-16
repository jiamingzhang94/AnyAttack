### Evaluation Adversarial Images on MiniGPT4

Evaluate the trained models on MiniGPT4 after generating adversarial images by AnyAttack:

1.**MiniGPT-4 (LLaMA-2 Chat 7B)**
```bash
bash bash_minigpt4.sh
```

Replace `"PROJECT_PATH"` with project path.Download the checkpoint from [here](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view) and fill the ckpt_path in `bash_minigpt4.sh`.

2.**MiniGPT-v2** 

```bash
bash bash_minigpt4v2.sh
```

Replace `"PROJECT_PATH"` with project path.Download the checkpoint from [here](https://drive.google.com/file/d/1aVbfW7nkCSYx99_vCRyP1sOlQiWVSnAl/view) and fill the ckpt_path in `bash_minigpt4v2.sh`.

Additional implementation details of MiniGPT-4 can be found [here](https://github.com/Vision-CAIR/MiniGPT-4?tab=readme-ov-file).
