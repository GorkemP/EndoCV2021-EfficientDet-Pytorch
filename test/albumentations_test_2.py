# Created by Gorkem Polat at 23.02.2021
# contact: polatgorkem@gmail.com


import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A

set_name = "val"
image_folder_path = os.path.join("../datasets/polyps", set_name)
annotations_path = "../datasets/polyps/annotations/instances_" + set_name + ".json"

f = open(annotations_path)
annotations = json.load(f)
class_names = [x["name"] for x in annotations["categories"]]

colors_for_classes = ["yellow"]


def draw_bbox_on_image(image, annotations):
    # fig, ax = plt.subplots(figsize=(15, 15))
    fig, ax = plt.subplots()
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

    plt.tight_layout()
    plt.axis("off")
    plt.show()


def draw_bbox_on_image_augmented(image, annotations, categories):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i in range(len(annotations)):
        rect = patches.Rectangle((annotations[i][0], annotations[i][1]),
                                 annotations[i][2],
                                 annotations[i][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[categories[i] - 1],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(annotations[i][0],
                 annotations[i][1] - 3,
                 class_names[categories[i] - 1],
                 color=colors_for_classes[categories[i] - 1])

    plt.tight_layout()
    plt.axis("off")
    plt.show()


def draw_bbox_on_image_augmented_solo(image, annotations):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i in range(len(annotations)):
        rect = patches.Rectangle((annotations[i][0], annotations[i][1]),
                                 annotations[i][2],
                                 annotations[i][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[0],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(annotations[i][0],
                 annotations[i][1] - 3,
                 class_names[0],
                 color=colors_for_classes[0])

    plt.tight_layout()
    plt.axis("off")
    plt.show()


image_id = 21  # 18 => double polyp
image_name = annotations["images"][image_id]["file_name"]
original_file_name = annotations["images"][image_id]["original_file_name"]
image_annotations = [x for x in annotations["annotations"] if x["image_id"] == image_id]

image = cv2.imread(os.path.join(image_folder_path, image_name))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
draw_bbox_on_image(image, image_annotations)

bboxes = []
for object in image_annotations:
    bbox = object["bbox"]
    bbox.append(object["category_id"] - 1)
    bboxes.append(bbox)

print(original_file_name)
print("bboxes: " + str(len(bboxes)))

bboxes_np = np.array([np.array(xi) for xi in bboxes])

transform = A.Compose([
    # A.Rotate(limit=(-180, 180), p=1, border_mode=cv2.BORDER_CONSTANT)
    A.HorizontalFlip(p=1)
], bbox_params=A.BboxParams(format="coco", min_visibility=0.25))

transformed = transform(image=image, bboxes=bboxes_np)
transformed_image = transformed["image"]
transformed_bboxes = transformed["bboxes"]

draw_bbox_on_image_augmented_solo(transformed_image, transformed_bboxes)
