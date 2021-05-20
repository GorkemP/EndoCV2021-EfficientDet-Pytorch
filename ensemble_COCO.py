# Created by Gorkem Polat at 14.03.2021
# contact: polatgorkem@gmail.com
import glob
import os
import json
import shutil
from ensemble_boxes import *
import argparse

# parser = argparse.ArgumentParser(description='EndoCV2021: inference on test set, by Ece Isik Polat')
# parser.add_argument("-it", "--iou_threshold", type=float, default=0.3)
# args = parser.parse_args()

weights = [1, 1, 1, 1]
# iou_thr = args.iou_threshold
iou_thr = 0.4
skip_box_thr = 0.0001

predicted_path_list = ["test_bbox_results_0.json", "test_bbox_results_1.json", "test_bbox_results_2.json",
                       "test_bbox_results_3.json"]
ground_truth_path = "datasets/polyps_paper/annotations/instances_test.json"


def calculate_normalized_voc_given_json_path(predicted_path, ground_truth_path):
    f1 = open(predicted_path)
    json_dict = json.load(f1)

    f2 = open(ground_truth_path)
    originals = json.load(f2)

    organized_json_dict = []
    organized_counter = 0

    for i in range(len(json_dict)):
        image_id = json_dict[i]["image_id"]
        image_width = originals["images"][image_id]["width"]
        image_height = originals["images"][image_id]["height"]

        x1 = json_dict[i]["bbox"][0]
        y1 = json_dict[i]["bbox"][1]
        w = json_dict[i]["bbox"][2]
        h = json_dict[i]["bbox"][3]
        x2 = x1 + w
        y2 = y1 + h

        if x2 > image_width:
            x2 = image_width
        if y2 > image_height:
            y2 = image_height

        voc = [x1, y1, x2, y2]
        normalized = [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
        json_dict[i].update({"voc": voc})
        json_dict[i].update({"normalized": normalized})

        if ((x1 < image_width) & (y1 < image_height) & (y2 > y1) & (x2 > x1)):
            organized_json_dict.append(json_dict[i])
            organized_counter = organized_counter + 1
    return organized_json_dict


def get_original_images_id_list(ground_truth_path):
    f = open(ground_truth_path)
    json_dict = json.load(f)
    original_images_ids = []
    for org_img in json_dict["images"]:
        original_images_ids.append(org_img["id"])
    return original_images_ids


original_images_ids = get_original_images_id_list(ground_truth_path)


def get_enseble_results(predicted_path_list, ground_truth_path):
    f_gt = open(ground_truth_path)
    gt_dict = json.load(f_gt)

    original_images_id_list = get_original_images_id_list(ground_truth_path)
    fusion_dict = []

    for image_id in original_images_id_list:
        boxes_list = []
        scores_list = []
        labels_list = []

        for json_path in predicted_path_list:

            json_dict = calculate_normalized_voc_given_json_path(json_path, ground_truth_path)

            image_annotations = [x for x in json_dict if x["image_id"] == image_id]
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

        image_width = gt_dict["images"][image_id]["width"]
        image_height = gt_dict["images"][image_id]["height"]

        annotation_counter = 0

        for i in range(len(scores)):
            x1 = int(boxes[i, 0] * image_width)
            y1 = int(boxes[i, 1] * image_height)
            x2 = int(boxes[i, 2] * image_width)
            y2 = int(boxes[i, 3] * image_height)

            object_width = x2 - x1
            object_height = y2 - y1

            annotation_dict = {}
            annotation_dict["image_id"] = image_id
            annotation_dict["category_id"] = 1
            annotation_dict["score"] = scores[i].astype(float)
            annotation_dict["bbox"] = [x1, y1, object_width, object_height]
            fusion_dict.append(annotation_dict)
            annotation_counter += 1

    with open("ensemble.json", "w") as outfile:
        json.dump(fusion_dict, outfile)


get_enseble_results(predicted_path_list, ground_truth_path)
