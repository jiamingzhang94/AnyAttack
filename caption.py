import os
import argparse
import random
import re

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from lavis.common.registry import registry

from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.coco_caption_datasets import COCOCapEvalDataset
from lavis.processors.blip_processors import BlipImageBaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from lavis.processors.clip_processors import _convert_to_rgb
import lavis.common.utils as utils
import warnings
from lavis.processors.randaugment import RandomAugment
from lavis.datasets.datasets.coco_caption_datasets import NoCapsEvalDataset
from lavis_tool.caption import CaptionTask
# from lavis.tasks.captioning import CaptionTask

# from lavis.processors.blip_processors import BlipCaptionProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg_path", help="path to configuration file.")
    parser.add_argument("--cache_path",  help="path to dataset cache")
    parser.add_argument("--json_path",help="test data path")
    parser.add_argument("--gt_path", help="path to groung truth")
    parser.add_argument("--image_path", help="path to image dataset")
    parser.add_argument("--output_dir", help="path where to save result")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
            self, image_size=384, transform=None, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(min_scale, max_scale),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(
                        2,
                        5,
                        isPIL=True,
                        augs=[
                            "Identity",
                            "AutoContrast",
                            "Brightness",
                            "Sharpness",
                            "Equalize",
                            "ShearX",
                            "ShearY",
                            "TranslateX",
                            "TranslateY",
                            "Rotate",
                        ],
                    ),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            self.transform = transform

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, transform=None, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            self.transform = transform

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


class ClipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
            self, image_size=224, transform=None, mean=None, std=None, min_scale=0.9, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(min_scale, max_scale),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    _convert_to_rgb,
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            self.transform = transform

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.9)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class ClipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, transform=None, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size),
                    _convert_to_rgb,
                    transforms.ToTensor(),
                    self.normalize,
                ]
            )
        else:
            self.transform = transform

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
        )


def _build_proc_from_cfg(cfg):
    return (
        registry.get_processor_class(cfg.name).from_config(cfg)
        if cfg is not None
        else None
    )


def build(cfg, transform=None):
    """
    Create by split datasets inheriting torch.utils.data.Datasets.

    # build() can be dataset-specific. Overwrite to customize.
    """
    try:
        image_size = cfg.config['preprocess']['vis_processor']['eval']['image_size']
    except:
        image_size = 384

    config = cfg.config['datasets']
    # self.build_processors()
    text_processor_dict = {'name': 'blip_caption'}

    vis_processors = {'train': BlipImageTrainProcessor(image_size=image_size, transform=transform),
                      'eval': BlipImageEvalProcessor(image_size=image_size, transform=transform)}
    text_processors = {'train': registry.get_processor_class('blip_caption').from_config({'name': 'blip_caption'}),
                       'eval': registry.get_processor_class('blip_caption').from_config({'name': 'blip_caption'})}
    retrieval_datasets_keys = list(config.keys())
    build_info = config[retrieval_datasets_keys[0]]['build_info']
    # build_info = config.build_info

    ann_info = build_info['annotations']
    data_type = config[retrieval_datasets_keys[0]]['data_type']
    vis_info = build_info[data_type]

    datasets = dict()
    for split in ann_info.keys():
        if split not in ["train", "val", "test"]:
            continue

        is_train = split == "train"

        # processors
        vis_processor = (
            vis_processors["train"]
            if is_train
            else vis_processors["eval"]
        )
        text_processor = (
            text_processors["train"]
            if is_train
            else text_processors["eval"]
        )
        # annotation path
        ann_paths = ann_info.get(split).storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        abs_ann_paths = []
        for ann_path in ann_paths:
            if not os.path.isabs(ann_path):
                ann_path = utils.get_cache_path(ann_path)
            abs_ann_paths.append(ann_path)
        ann_paths = abs_ann_paths

        # visual data storage path
        vis_path = vis_info.storage

        if not os.path.isabs(vis_path):
            # vis_path = os.path.join(utils.get_cache_path(), vis_path)
            vis_path = utils.get_cache_path(vis_path)

        if not os.path.exists(vis_path):
            warnings.warn("storage path {} does not exist.".format(vis_path))

        # create datasets
        if retrieval_datasets_keys[0]=="nocaps":
            dataset_cls = NoCapsEvalDataset
        else:
            dataset_cls = CaptionDataset if is_train else COCOCapEvalDataset
        datasets[split] = dataset_cls(
            vis_processor=vis_processor,
            text_processor=text_processor,
            ann_paths=ann_paths,
            vis_root=vis_path,
        )
    datasets_retrieval = {retrieval_datasets_keys[0]: datasets}
    return datasets_retrieval



def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    args = parse_args()
    registry.mapping["paths"]["cache_root"] = args.cache_path
    job_id = now()

    cfg = Config(args)
    registry.mapping["paths"]["coco_gt"]=""
    if args.image_path:
        cfg.config['datasets'][list(cfg.config['datasets'].keys())[0]]['build_info']['images'][
            'storage'] = args.image_path
    if args.output_dir:
        cfg.config['run']['output_dir'] = args.output_dir
    if args.json_path:
        dataset_name = list(cfg.config['datasets'].keys())[0]
        cfg.config['datasets'][dataset_name]['build_info']['annotations']['test']['storage'] = args.json_path
    if args.gt_path:
        registry.mapping["paths"]["coco_gt"]=args.gt_path
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    if cfg.config['datasets']=="nocaps":
        task = tasks.setup_task(cfg)
    else:
        task = CaptionTask.setup_task(cfg=cfg)

    try:
        image_size = cfg.config['preprocess']['vis_processor']['eval']['image_size']
    except:
        image_size = 384
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize
        ]
    )
    datasets = build(cfg, transform=transform)
    # datasets = build(cfg, transform=None)
    # datasets = task.build_datasets(cfg)


    model = task.build_model(cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    output_dir = os.path.join(cfg.run_cfg["output_dir"], args.image_path[args.image_path.rfind("/") + 1:])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    registry.mapping["paths"]["output_dir"] = output_dir
    registry.mapping["paths"]["result_dir"] = output_dir

    runner.evaluate(skip_reload=True)

if __name__ == "__main__":
    main()