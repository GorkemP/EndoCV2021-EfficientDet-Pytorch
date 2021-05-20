# Created by Gorkem Polat at 6.03.2021
# contact: polatgorkem@gmail.com

import glob
import os
import json
import shutil
from ensemble_boxes import *
import argparse

parser = argparse.ArgumentParser(description='EndoCV2021: inference on test set, by Ece Isik Polat')
parser.add_argument("-it", "--iou_threshold", type=float, default=0.3)
args = parser.parse_args()

weights = [1, 1, 1, 1]
iou_thr = args.iou_threshold
skip_box_thr = 0.0001
# sigma = 0.1

target_folder_name = "ensemble_result"
if os.path.isdir(target_folder_name):
    shutil.rmtree(target_folder_name)
os.makedirs(target_folder_name)


def calculate_normalized_voc_given_json_path(json_path):
    f = open(json_path)
    json_dict = json.load(f)

    organized_json_dict = {}
    organized_json_dict["categories"] = json_dict["categories"]
    organized_json_dict["images"] = json_dict["images"]
    organized_json_dict["annotations"] = []
    organized_counter = 0

    original_images_ids = []
    for org_img in json_dict["images"]:
        original_images_ids.append(org_img["id"])
    for image_id in original_images_ids:
        image_width = json_dict["images"][image_id]["width"]
        image_height = json_dict["images"][image_id]["height"]
        image_annotations = [x for x in json_dict["annotations"] if x["image_id"] == image_id]

        for ann in image_annotations:
            x1 = ann["bbox"][0]
            y1 = ann["bbox"][1]
            w = ann["bbox"][2]
            h = ann["bbox"][3]
            x2 = x1 + w
            y2 = y1 + h

            if x2 > image_width:
                x2 = image_width
            if y2 > image_height:
                y2 = image_height

            voc = [x1, y1, x2, y2]
            normalized = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
            ann.update({"voc": voc})
            ann.update({"normalized": normalized})

            if ((x1 < image_width) & (y1 < image_height) & (y2 > y1) & (x2 > x1)):
                organized_json_dict["annotations"].append(ann)
                organized_counter = organized_counter + 1
    return organized_json_dict


## burada bütün modellerin altındaki jsonlar voca çevrilip normalize ediliyor
folders = ["model_1", "model_2", "model_3", "model_4"]

for folder in folders:
    path = os.path.join(folder, "")
    jsonlist = glob.glob(path + "*.json")
    for json_path in jsonlist:
        file_name = json_path[8:-5] + "_org"
        organized_json_dict = calculate_normalized_voc_given_json_path(json_path)

        write_path = os.path.join(folder, file_name)
        with open(write_path + ".json", "w") as outfile:
            json.dump(organized_json_dict, outfile)


def get_original_images_id_list(file_name):
    json_path = os.path.join("model_1", file_name)
    f = open(json_path)
    json_dict = json.load(f)
    original_images_ids = []
    for org_img in json_dict["images"]:
        original_images_ids.append(org_img["id"])
    return original_images_ids


def get_enseble_results(file):
    folder_list = ["model_1", "model_2", "model_3", "model_4"]

    original_images_id_list = get_original_images_id_list(file_name=file)
    fusion_dict = {}
    fusion_dict["categories"] = []
    fusion_dict["images"] = []
    fusion_dict["annotations"] = []

    for image_id in original_images_id_list:
        boxes_list = []
        scores_list = []
        labels_list = []

        for fold in folder_list:

            json_path = os.path.join(fold, file)
            f = open(json_path)

            json_dict = json.load(f)

            image_annotations = [x for x in json_dict["annotations"] if x["image_id"] == image_id]
            bb = []
            scr = []
            lbl = []
            for ann in image_annotations:

                for j in range(4):
                    if (ann["normalized"][j] < 0):
                        print(json_path, ann["id"], image_id, ann["normalized"][j])
                    if (ann["normalized"][j] > 1):
                        print(json_path, ann["id"], image_id, ann["normalized"][j])
                bb.append(ann["normalized"])
                scr.append(ann["score"])
                lbl.append(1)
            boxes_list.append(bb)
            scores_list.append(scr)
            labels_list.append(lbl)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        image_width = json_dict["images"][image_id]["width"]
        image_height = json_dict["images"][image_id]["height"]

        annotation_counter = 0

        for i in range(len(scores)):
            x1 = int(boxes[i, 0] * image_width)
            y1 = int(boxes[i, 1] * image_height)
            x2 = int(boxes[i, 2] * image_width)
            y2 = int(boxes[i, 3] * image_height)

            object_width = x2 - x1
            object_height = y2 - y1

            annotation_dict = {}
            annotation_dict["id"] = annotation_counter
            annotation_dict["image_id"] = image_id
            annotation_dict["category_id"] = 1
            annotation_dict["iscrowd"] = 0
            annotation_dict["area"] = object_width * object_height
            annotation_dict["bbox"] = [x1, y1, object_width, object_height]
            annotation_dict["score"] = scores[i].astype(float)
            fusion_dict["annotations"].append(annotation_dict)
            annotation_counter += 1

    fusion_dict["categories"] = json_dict["categories"]
    fusion_dict["images"] = json_dict["images"]

    ensemble_file = file[:-9] + ".json"
    with open(os.path.join(target_folder_name, ensemble_file), "w") as outfile:
        json.dump(fusion_dict, outfile)


new_file_list = ["EndoCV_DATA1_org.json", "EndoCV_DATA2_org.json", "EndoCV_DATA3_org.json"]
for nf in new_file_list:
    get_enseble_results(nf)
