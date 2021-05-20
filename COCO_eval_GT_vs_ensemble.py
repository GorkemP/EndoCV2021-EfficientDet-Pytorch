# Created by Gorkem Polat at 15.03.2021
# contact: polatgorkem@gmail.com

# Created by Gorkem Polat at 11.03.2021
# contact: polatgorkem@gmail.com


import json
import os
import glob
import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


if __name__ == '__main__':
    # VAL_GT = f'datasets/polyps_paper/annotations/instances_test.json'
    # target_json = "ensemble.json"
    # VAL_GT = '/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/kvasir_seg_COCO.json'
    # target_json = "ensemble_kvasir.json"
    # VAL_GT = '/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/kvasir_seg_test_COCO.json'
    # target_json = "ensemble_kvasir_test.json"
    VAL_GT = 'datasets/polyps_all_centers/annotations/instances_test.json'
    target_json = "ensemble_kvasir_on_endocv_center.json"
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()

    coco_result = _eval(coco_gt, image_ids, target_json)
