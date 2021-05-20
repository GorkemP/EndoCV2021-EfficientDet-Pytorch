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

project_name = "polyps_paper"
efficientdet_version = 1
weights_file = "trained_weights/efficientdet-d" + str(efficientdet_version) + "_best_105.pth"
VAL_GT = f'datasets/{project_name}/annotations/instances_test.json'
VAL_IMGS = f"datasets/{project_name}/test"
target_json_name = "test_bbox_results_" + str(efficientdet_version) + ".json"
conf_threshold = 0.2
nms_threshold = 0.5
use_cuda = True
gpu = 0

print(f'running coco-style evaluation on project {project_name}, weights {weights_file}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(img_path, image_info['file_name'])

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[efficientdet_version],
                                                         mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id'   : image_id,
                    'category_id': label + 1,
                    'score'      : float(score),
                    'bbox'       : box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    if os.path.exists(target_json_name):
        os.remove(target_json_name)
    json.dump(results, open(target_json_name, 'w'), indent=4)


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
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()

    model = EfficientDetBackbone(compound_coef=efficientdet_version, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

    evaluate_coco(VAL_IMGS, image_ids, coco_gt, model, conf_threshold)

    coco_result = _eval(coco_gt, image_ids, target_json_name)
