# Created by Gorkem Polat at 10.03.2021
# contact: polatgorkem@gmail.com

import os
import glob
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

root_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG"
source_images_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/images"
annotations_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/kvasir_bboxes.json"

f = open(annotations_path)
annotations = json.load(f)

COCO_json = {}
COCO_json["categories"] = []
COCO_json["images"] = []
COCO_json["annotations"] = []

category = {}
category["id"] = 1
category["name"] = "polyp"
category["supercategory"] = "None"
COCO_json["categories"].append(category)

image_paths = glob.glob(os.path.join(source_images_path, "*.jpg"))

image_counter = 0
annotation_counter = 0
for image_path in tqdm(image_paths):
    image_name = image_path.split("/")[-1][:-4]

    image_annotation = annotations[image_name]
    if len(image_annotation["bbox"]) > 0:
        for bbox_annotation in image_annotation["bbox"]:
            if bbox_annotation["label"] == "polyp":
                bbox_width = bbox_annotation["xmax"] - bbox_annotation["xmin"]
                bbox_height = bbox_annotation["ymax"] - bbox_annotation["ymin"]

                annotation_dict = {}
                annotation_dict["id"] = annotation_counter
                annotation_dict["image_id"] = image_counter
                annotation_dict["category_id"] = 1
                annotation_dict["iscrowd"] = 0
                annotation_dict["area"] = bbox_width * bbox_height
                annotation_dict["bbox"] = [bbox_annotation["xmin"], bbox_annotation["ymin"], bbox_width, bbox_height]
                COCO_json["annotations"].append(annotation_dict)
                annotation_counter += 1
            else:
                print("Different object in " + image_name)

        image_dict = {}
        image_dict["id"] = image_counter
        image_dict["file_name"] = image_name + ".jpg"
        image_dict["width"] = image_annotation["width"]
        image_dict["height"] = image_annotation["height"]
        COCO_json["images"].append(image_dict)
        image_counter += 1

    else:
        print("No object for " + image_name)

with open(os.path.join(root_path, "kvasir_seg_COCO.json"), "w") as outfile:
    json.dump(COCO_json, outfile)
