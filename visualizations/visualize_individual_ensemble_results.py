# Created by Gorkem Polat at 5.03.2021
# contact: polatgorkem@gmail.com

import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

set_name = "val"
image_folder_path = "../datasets/polyps/test_set_trial/EndoCV_DATA1"
annotation1 = "../model_1/EndoCV_DATA3.json"
annotation2 = "../model_2/EndoCV_DATA3.json"
annotation3 = "../model_3/EndoCV_DATA3.json"

annotation_ensemble = "../ensemble_result/EndoCV_DATA3.json"

f = open(annotation1)
annotations1 = json.load(f)

f = open(annotation2)
annotations2 = json.load(f)

f = open(annotation3)
annotations3 = json.load(f)

f = open(annotation_ensemble)
annotations_ensemble = json.load(f)

colors_for_classes = ["yellow", "cyan", "magenta", "lime"]


def draw_bbox_on_image_individual(image, annotations1, annotations2, annotations3):
    image = image[:, :, ::-1]
    fig, ax = plt.subplots()
    ax.imshow(image)

    for object in annotations1:
        rect = patches.Rectangle((object["bbox"][0], object["bbox"][1]),
                                 object["bbox"][2],
                                 object["bbox"][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[0],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(object["bbox"][0],
                 object["bbox"][1] - 3,
                 "{:.2f}".format(object["score"]),
                 color=colors_for_classes[0])

    for object in annotations2:
        rect = patches.Rectangle((object["bbox"][0], object["bbox"][1]),
                                 object["bbox"][2],
                                 object["bbox"][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[1],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(object["bbox"][0],
                 object["bbox"][1] - 3,
                 "{:.2f}".format(object["score"]),
                 color=colors_for_classes[1])

    for object in annotations3:
        rect = patches.Rectangle((object["bbox"][0], object["bbox"][1]),
                                 object["bbox"][2],
                                 object["bbox"][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[2],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(object["bbox"][0],
                 object["bbox"][1] - 3,
                 "{:.2f}".format(object["score"]),
                 color=colors_for_classes[2])

    plt.tight_layout()
    plt.axis("off")
    plt.show()


def draw_bbox_on_image_ensemble(image, annotations_ensemble):
    image = image[:, :, ::-1]
    fig, ax = plt.subplots()
    ax.imshow(image)

    for object in annotations_ensemble:
        rect = patches.Rectangle((object["bbox"][0], object["bbox"][1]),
                                 object["bbox"][2],
                                 object["bbox"][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[3],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(object["bbox"][0],
                 object["bbox"][1] - 3,
                 "{:.2f}".format(object["score"]),
                 color=colors_for_classes[3])

    plt.tight_layout()
    plt.axis("off")
    plt.show()


image_id = random.randint(0, 270)
# image_id = 109
image_name = annotations1["images"][image_id]["file_name"]

image_annotations1 = [x for x in annotations1["annotations"] if x["image_id"] == image_id]
image_annotations2 = [x for x in annotations2["annotations"] if x["image_id"] == image_id]
image_annotations3 = [x for x in annotations3["annotations"] if x["image_id"] == image_id]

image_annotations_ensemble = [x for x in annotations_ensemble["annotations"] if x["image_id"] == image_id]

image = cv2.imread(os.path.join(image_folder_path, image_name))
print(image_name)
draw_bbox_on_image_individual(image, image_annotations1, image_annotations2, image_annotations3)
draw_bbox_on_image_ensemble(image, image_annotations_ensemble)
