import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
# import gradio as gr
from PIL import Image
from transformers import StoppingCriteriaList
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub,Conversation, SeparatorStyle
from eval import eval_caption
device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"
conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

def setup_seeds_another(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
# ------------------------------------------------------------------ #

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

def initialize_model(cfg):
    # ========================================
    #             Model Initialization
    # ========================================
    setup_seeds_another(42)

    model_config = cfg.model_cfg
    model_config.do_sample = False
    # model_config.device_8bit = os.environ["CUDA_VISIBLE_DEVICES"]
    model_cls = registry.get_model_class(model_config.arch)
    # model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model = model_cls.from_config(model_config).to(device)
    if cfg.model_cfg.arch=='minigpt_v2':
        CONV_VISION = Conversation(
            system="",
            roles=(r"<s>[INST] ", r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="",
        )
    else:
        CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    # stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stop_words_ids = [torch.tensor(ids).to(device=device) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    chat = Chat(model, vis_processor, device=device, stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    return chat, CONV_VISION

def chat_with_image_path_and_question(chat, CONV_VISION, image_path, query):
    # ========================================
    #                Chatting
    # ========================================

    image = Image.open(image_path).convert('RGB')

    # print("Image loaded.")
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list)
    # print("Chat loaded.")
    chat.encode_img(img_list)
    # print("Chat encode_img.")
    chat.ask(query, chat_state)
    # print("Chat ask.")

    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              max_new_tokens=300,
                              max_length=2000)[0]
    print("Answer:", llm_message)
    return llm_message



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # minigpt-4
    parser.add_argument("--cfg_path", default="minigpt4_eval.yaml")
    parser.add_argument("--data_path", default="coco/annotations/coco_karpathy_test.json", type=str)
    parser.add_argument("--image_path", default="ms_coco", type=str)
    parser.add_argument("--gt_path", default="coco_karpathy_test_gt.json")
    parser.add_argument("--llama_path",default='Llama-2-7b-chat-hf')
    parser.add_argument("--ckpt_path",help="path to the adapter path")
    parser.add_argument("--output_path", default="outputs/result.json", type=str)
    parser.add_argument("--prompt",default="Describe this image in one short sentence only.",type=str)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    print("output_path:", args.output_path)
    print(f"Loading MiniGPT-4 models..")
    cfg = Config(args)
    if args.llama_path:
        cfg.config["model"]["llama_model"]=args.llama_path
    if args.ckpt_path:
        cfg.config["model"]["ckpt"]=args.ckpt_path

    chat, CONV_VISION=initialize_model(cfg)
    print(f"Done")

    with open(args.data_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    images_path = args.image_path

    dir_name = os.path.dirname(args.output_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    result = []
    for idx,i in enumerate(data):
        image_path=os.path.join(images_path,i["image"])
        print(image_path)
        caption = chat_with_image_path_and_question(chat, CONV_VISION, image_path, args.prompt)
        try:
            image_id = int(i["image"].split("_")[-1].split(".jpg")[0])
        except:
            image_id = int(i["image"].split(".png")[0])
        result.append(
            {
                "caption":caption,
                "image_id":image_id
            }
        )
        print(idx,"-"*100)


        with open(args.output_path,"w",encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=4)

    eval_caption(args.gt_path,args.output_path)

