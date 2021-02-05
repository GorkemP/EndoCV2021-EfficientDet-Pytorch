# Created by Gorkem Polat at 5.02.2021
# contact: polatgorkem@gmail.com

import os
import torch
from torch.backends import cudnn
import random
import glob
import json
import yaml

from backbone import EfficientDetBackbone
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


def draw_bboxes_on_image(image, annotations, class_names):
    image = image[:, :, ::-1]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)

    for i in range(len(out[0]["rois"])):
        rect = patches.Rectangle((annotations[0]["rois"][i][0], annotations[0]["rois"][i][1]),
                                 annotations[0]["rois"][i][2] - annotations[0]["rois"][i][0],
                                 annotations[0]["rois"][i][3] - annotations[0]["rois"][i][1],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[annotations[0]["class_ids"][i]],
                                 facecolor='none')
        ax.add_patch(rect)
        plt.text(annotations[0]["rois"][i][0],
                 annotations[0]["rois"][i][1] - 3,
                 class_names[annotations[0]["class_ids"][i]] + " " + "{:.2f}".format(annotations[0]["scores"][i]),
                 color=colors_for_classes[annotations[0]["class_ids"][i]])

    plt.axis("off")
    plt.show()


def draw_groundtruth_bboxes_on_image(image, annotations, class_names):
    image = image[:, :, ::-1]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)

    for object in annotations:
        rect = patches.Rectangle((object["bbox"][0], object["bbox"][1]),
                                 object["bbox"][2],
                                 object["bbox"][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[object["category_id"] - 1],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(object["bbox"][0],
                 object["bbox"][1] - 3,
                 class_names[object["category_id"] - 1],
                 color=colors_for_classes[object["category_id"] - 1])

    plt.axis("off")
    plt.show()


only_prediction = True
set_name = "val"
project_name = "birdview_vehicles"
compound_coef = 0
force_input_size = None  # set None to use default size
img_paths = glob.glob("../datasets/" + project_name + "/" + set_name + "/*.jpg")
annotations_path = "../datasets/" + project_name + "/annotations/instances_" + set_name + ".json"
f = open(annotations_path)
annotations = json.load(f)

img_path = img_paths[random.randint(0, len(img_paths))]
img_name = img_path.split("/")[-1]

image_id = [x["id"] for x in annotations["images"] if x["file_name"] == img_name]

image_gt_annotations = [x for x in annotations["annotations"] if x["image_id"] == image_id[0]]

weight_file = f"../logs/birdview_vehicles/efficientdet-d{compound_coef}_best.pth"

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

params = yaml.safe_load(open(f'../projects/{project_name}.yml'))
obj_list = params['obj_list']

viridis = cm.get_cmap('viridis', len(obj_list))
colors_for_classes = [viridis(x) for x in range(len(obj_list))]

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef,
                             num_classes=len(obj_list),
                             ratios=eval(params['anchors_ratios']),
                             scales=eval(params['anchors_scales']))

model.load_state_dict(torch.load(weight_file))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

out = invert_affine(framed_metas, out)
image = cv2.imread(os.path.join(img_path))

draw_bboxes_on_image(image, out, obj_list)
# plt.close()
draw_groundtruth_bboxes_on_image(image, image_gt_annotations, obj_list)
