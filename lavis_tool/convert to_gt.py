import argparse
import json
import os
from tqdm import tqdm


def convert_to_gt(gt_path, select_num, output_path):
    with open(gt_path, "r", encoding='utf-8') as f:
        gt = json.load(f)

    completed_annotations = []  # 新列表用于保存已完成更新的注解
    for idx, image in enumerate(tqdm(gt["images"])):
        for j in gt["annotations"].copy():  # 遍历注解的副本
            if j["image_id"] == image["id"]:
                j["image_id"] = idx
                completed_annotations.append(j)  # 保存到新列表
                gt["annotations"].remove(j)  # 从原列表中移除

        image['id'] = idx
    gt['images'] = gt["images"][:select_num]
    new_gt={ "annotations": completed_annotations,"images": gt["images"][:select_num]}
    # save_path =os.path.join(os.path.dirname(gt_path),"adv_gt.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_gt, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # minigpt-4
    # parser.add_argument("--result_path", default="/new_data/yifei2/junhong/text_guide_attack/compared_methods/minigpt4/minigpt_temp.json", help="path to model caption "
    #                                                                                       "result file.")
    parser.add_argument("--gt_path", default='/home/dycpu6_8tssd1/jmzhang/datasets/lavis/coco_gt/coco_karpathy_test_gt.json',
                        help="path to the ground truth file.")
    parser.add_argument("--select_num", default=1000, help="select the number of ground truth to be used.")
    parser.add_argument("--output_path", default='../json/coco_caption_test_gt_adv.json')
    args = parser.parse_args()
    convert_to_gt(args.gt_path,args.select_num,args.output_path)