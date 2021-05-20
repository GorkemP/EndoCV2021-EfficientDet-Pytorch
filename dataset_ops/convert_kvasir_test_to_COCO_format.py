# Created by Gorkem Polat at 15.03.2021
# contact: polatgorkem@gmail.com

import os
import glob
import cv2
import json
import shutil
from tqdm import tqdm

target_set_name = "test"
source_files_path = os.path.join(
        "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG", target_set_name)
GT_annotation = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/kvasir_seg_COCO.json"

classes = ["polyp"]

f = open(GT_annotation)
all_annotations = json.load(f)

image_paths = glob.glob(os.path.join(source_files_path, "*.jpg"))

annotations = {}
annotations["categories"] = []
annotations["images"] = []
annotations["annotations"] = []
for id, class_name in enumerate(classes):
    category = {}
    category["id"] = id + 1
    category["name"] = class_name
    category["supercategory"] = "None"
    annotations["categories"].append(category)

image_counter = 0
annotation_counter = 0
for image_path in tqdm(image_paths):
    image_name = image_path.split("/")[-1]
    image_id = [x for x in all_annotations["images"] if x["file_name"] == image_name]
    image_id = image_id[0]["id"]
    bbox_annotations = [x for x in all_annotations["annotations"] if x["image_id"] == image_id]

    try:
        if len(bbox_annotations) > 0 and not ("\n" in bbox_annotations):
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            for bbox_annotation in bbox_annotations:
                annotation_dict = {}
                annotation_dict["id"] = annotation_counter
                annotation_dict["image_id"] = image_counter
                annotation_dict["category_id"] = bbox_annotation["category_id"]
                annotation_dict["iscrowd"] = 0
                annotation_dict["area"] = bbox_annotation["area"]
                annotation_dict["bbox"] = bbox_annotation["bbox"]
                annotations["annotations"].append(annotation_dict)
                annotation_counter += 1

            image_dict = {}
            image_dict["id"] = image_counter
            image_dict["file_name"] = str(image_counter) + ".jpg"
            image_dict["original_file_name"] = image_name
            image_dict["width"] = width
            image_dict["height"] = height
            annotations["images"].append(image_dict)
            image_counter += 1
        else:
            print("No bbox information for: " + image_name)
    except Exception as e:
        print("image path:" + image_path)
        print(str(e.message))
        break

with open("/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/kvasir_seg_test_COCO.json",
          "w") as outfile:
    json.dump(annotations, outfile)
