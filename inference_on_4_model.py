# Created by Gorkem Polat at 6.03.2021
# contact: polatgorkem@gmail.com

import os
import cv2
import json
import torch
from torch.backends import cudnn
import glob
import yaml
import numpy as np
import shutil
from backbone import EfficientDetBackbone
import argparse

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

parser = argparse.ArgumentParser(description='EndoCV2021: inference on test set, by Gorkem Polat')
parser.add_argument("-if", "--image_folder", type=str, default="datasets/polyps_deneme")
parser.add_argument("-ct", "--confidence_threshold", type=float, default=0.1)
parser.add_argument("-it", "--iou_threshold", type=float, default=0.1)

parser.add_argument("-c1", "--configuration1", type=int, default=0)
parser.add_argument("-wf1", "--weight_file1", type=str, default="trained_weights/efficientdet-d0_best_91.pth")

parser.add_argument("-c2", "--configuration2", type=int, default=1)
parser.add_argument("-wf2", "--weight_file2", type=str, default="trained_weights/efficientdet-d1_best_89.pth")

parser.add_argument("-c3", "--configuration3", type=int, default=2)
parser.add_argument("-wf3", "--weight_file3", type=str, default="trained_weights/efficientdet-d2_best_94.pth")

parser.add_argument("-c4", "--configuration4", type=int, default=3)
parser.add_argument("-wf4", "--weight_file4", type=str, default="trained_weights/efficientdet-d3_best_90.pth")

parser.add_argument("-cu", "--cuda", type=str, default="T")
args = parser.parse_args()

test_set_path = args.image_folder
threshold = args.confidence_threshold
iou_threshold = args.iou_threshold

compound_coefs = [args.configuration1, args.configuration2, args.configuration3, args.configuration4]
weight_fileS = [args.weight_file1, args.weight_file2, args.weight_file3, args.weight_file4]

use_cuda_arg = args.cuda

force_input_size = None  # set None to use default size

if use_cuda_arg == "T":
    use_cuda = True
else:
    use_cuda = False
print("Use Cuda: " + str(use_cuda))

use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

directories = sorted(os.listdir(test_set_path))
result_folder_names = ["model_1", "model_2", "model_3", "model_4"]

if os.path.isdir(result_folder_names[0]):
    shutil.rmtree(result_folder_names[0])
os.makedirs(result_folder_names[0])

if os.path.isdir(result_folder_names[1]):
    shutil.rmtree(result_folder_names[1])
os.makedirs(result_folder_names[1])

if os.path.isdir(result_folder_names[2]):
    shutil.rmtree(result_folder_names[2])
os.makedirs(result_folder_names[2])

if os.path.isdir(result_folder_names[3]):
    shutil.rmtree(result_folder_names[3])
os.makedirs(result_folder_names[3])

for model_id in range(len(weight_fileS)):
    print("processing model: " + str(model_id + 1))

    compound_coef = compound_coefs[model_id]
    weight_file = weight_fileS[model_id]

    for directory in directories:
        print("---> processing folder: " + directory)

        directory_path = os.path.join(test_set_path, directory)

        img_paths = glob.glob(os.path.join(directory_path, "*.jpg"))
        img_paths = sorted(img_paths)
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

        annotations = {}
        annotations["categories"] = []
        annotations["images"] = []
        annotations["annotations"] = []

        category = {}
        category["id"] = 1
        category["name"] = "polyp"
        category["supercategory"] = "None"
        annotations["categories"].append(category)

        image_counter = 0
        annotation_counter = 0
        for img_path in img_paths:
            # print("processing: " + img_path.split("/")[-1])

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
                    (x1, y1, x2, y2) = out[i]['rois'][j].astype(int)
                    obj = obj_list[out[i]['class_ids'][j]]
                    score = float(out[i]['scores'][j])

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    object_width = x2 - x1
                    object_height = y2 - y1

                    annotation_dict = {}
                    annotation_dict["id"] = annotation_counter
                    annotation_dict["image_id"] = image_counter
                    annotation_dict["category_id"] = 1
                    annotation_dict["iscrowd"] = 0
                    annotation_dict["area"] = object_width * object_height
                    annotation_dict["bbox"] = [x1, y1, object_width, object_height]
                    annotation_dict["score"] = score
                    annotations["annotations"].append(annotation_dict)
                    annotation_counter += 1

            current_img = cv2.imread(img_path)
            height, width, _ = current_img.shape
            image_dict = {}
            image_dict["id"] = image_counter
            image_dict["file_name"] = img_path.split("/")[-1]
            image_dict["width"] = width
            image_dict["height"] = height
            annotations["images"].append(image_dict)
            image_counter += 1

        with open(os.path.join(result_folder_names[model_id], directory + ".json"), "w") as outfile:
            json.dump(annotations, outfile)
