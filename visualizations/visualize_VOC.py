# Created by Gorkem Polat at 28.02.2021
# contact: polatgorkem@gmail.com

import os
import glob
import cv2
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bbox_on_image(image, predictions, image_gt_annotations):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for prediction in predictions:
        information = prediction.split(' ')

        rect = patches.Rectangle((float(information[2]), float(information[3])),
                                 float(information[4]) - float(information[2]),
                                 float(information[5]) - float(information[3]),
                                 linewidth=2,
                                 edgecolor="yellow",
                                 facecolor="none")

        ax.add_patch(rect)
        plt.text(float(information[2]),
                 float(information[3]) - 10,
                 "polyp " + information[1],
                 color="yellow")

    for gt_annotation in image_gt_annotations:
        rect = patches.Rectangle((gt_annotation["bbox"][0], gt_annotation["bbox"][1]),
                                 gt_annotation["bbox"][2],
                                 gt_annotation["bbox"][3],
                                 linewidth=2,
                                 edgecolor="lime",
                                 facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    # plt.axis("off")
    plt.show()


parser = argparse.ArgumentParser(
        description='EndoCV2021: visualize image and their annotations from txt files, by Gorkem Polat')
parser.add_argument("-if", "--image_folder", type=str, default="../datasets/polyps/val")
parser.add_argument("-af", "--annotation_folder", type=str, default="../EndoCV_DATA2")
parser.add_argument("-gtf", "--gt_path", type=str, default="../datasets/polyps/annotations/instances_val.json")
args = parser.parse_args()

image_folder_path = args.image_folder
annotation_paths = args.annotation_folder
gt_annotations_path = args.gt_path

f = open(gt_annotations_path)
gt_annotations = json.load(f)

image_paths = glob.glob(os.path.join(image_folder_path, "*.jpg"))

for image_path in image_paths[:2]:
    image_name = image_path.split("/")[-1][:-4]
    annotation_path = os.path.join(annotation_paths, image_name + ".txt")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(annotation_path) as f:
        contents = f.readlines()

    image_gt_annotations = [x for x in gt_annotations["annotations"] if x["image_id"] == int(image_name)]

    draw_bbox_on_image(image, contents, image_gt_annotations)
