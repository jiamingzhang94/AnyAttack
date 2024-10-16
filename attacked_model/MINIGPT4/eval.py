import os
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import pycocoevalcap.spice as spice
import argparse

def eval_caption(gt_path,result_path):
    coco = COCO(gt_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path",default="caption.json",help="path to model caption result file.")
    parser.add_argument("--gt_path", default="coco_caption_test_gt_adv.json", help="path to the ground truth file.")
    args = parser.parse_args()
    eval_caption(args.gt_path,args.result_path)
