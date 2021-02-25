# Created by Gorkem Polat at 10.02.2021
# contact: polatgorkem@gmail.com

import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

set_name = "val"
image_folder_path = os.path.join("../datasets/polyps", set_name)
annotations_path = "../datasets/polyps/annotations/instances_" + set_name + ".json"

f = open(annotations_path)
annotations = json.load(f)
class_names = [x["name"] for x in annotations["categories"]]

colors_for_classes = ["yellow"]


def draw_bbox_on_image(image, annotations):
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

    plt.tight_layout()
    plt.axis("off")
    plt.show()


image_id = 10
image_name = annotations["images"][image_id]["file_name"]
original_file_name = annotations["images"][image_id]["original_file_name"]
image_annotations = [x for x in annotations["annotations"] if x["image_id"] == image_id]

image = cv2.imread(os.path.join(image_folder_path, image_name))
print(original_file_name)
draw_bbox_on_image(image, image_annotations)
