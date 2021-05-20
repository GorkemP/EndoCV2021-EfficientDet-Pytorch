# Created by Gorkem Polat at 26.02.2021
# contact: polatgorkem@gmail.com

import os
import torch
from torch.backends import cudnn
import glob
import yaml
import shutil
import numpy as np
from backbone import EfficientDetBackbone
import argparse

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

parser = argparse.ArgumentParser(description='EndoCV2021: inference on test set, by Gorkem Polat')
parser.add_argument("-if", "--image_folder", type=str, default="datasets/polyps/val")
parser.add_argument("-c", "--configuration", type=int, default=0)
parser.add_argument("-ct", "--confidence_threshold", type=float, default=0.1)
parser.add_argument("-it", "--iou_threshold", type=float, default=0.1)
parser.add_argument("-wf", "--weight_file", type=str, default="trained_weights/efficientdet-d0_best_51.pth")
parser.add_argument("-rf", "--result_folder", type=str, default="EndoCV_DATA1")
parser.add_argument("-cu", "--cuda", type=str, default="T")

args = parser.parse_args()

test_set_path = args.image_folder
compound_coef = args.configuration
weight_file = args.weight_file
inference_results_path = args.result_folder
threshold = args.confidence_threshold  # used for confidence
iou_threshold = args.iou_threshold  # Used for NMS
use_cuda_arg = args.cuda

force_input_size = None  # set None to use default size

if use_cuda_arg == "T":
    use_cuda = True
else:
    use_cuda = False
print("Use Cuda: "+str(use_cuda))

use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

if os.path.isdir(inference_results_path):
    shutil.rmtree(inference_results_path)
os.mkdir(inference_results_path)

img_paths = glob.glob(os.path.join(test_set_path, "*.jpg"))

params = yaml.safe_load(open(f'projects/polyps.yml'))
obj_list = params['obj_list']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef,
                             num_classes=len(obj_list),
                             ratios=eval(params['anchors_ratios']),
                             scales=eval(params['anchors_scales']))

model.load_state_dict(torch.load(weight_file, map_location="cpu"))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

for img_path in img_paths:
    img_name = img_path.split("/")[-1][:-4]
    txt_name = img_name + ".txt"
    target_txt_path = os.path.join(inference_results_path, txt_name)
    print("Processing: " + img_path.split("/")[-1])

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    results = []
    out = invert_affine(framed_metas, out)
    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        ori_imgs[i] = ori_imgs[i].copy()
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])

            prediction_txt = obj + " " + "{:.4f}".format(score) + " " + str(x1) + " " + str(y1) + " " + str(
                    x2) + " " + str(y2)
            results.append(prediction_txt)

    with open(target_txt_path, 'w') as f:
        for item in results:
            f.write("%s\n" % item)
