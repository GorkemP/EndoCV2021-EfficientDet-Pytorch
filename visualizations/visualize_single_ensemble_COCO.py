# Created by Gorkem Polat at 14.03.2021
# contact: polatgorkem@gmail.com

import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# image_folder_path = "../datasets/polyps_paper/test"
# GT_path = "../datasets/polyps_paper/annotations/instances_test.json"
# annotation1 = "../test_bbox_results_0.json"
# annotation2 = "../test_bbox_results_1.json"
# annotation3 = "../test_bbox_results_2.json"
# annotation4 = "../test_bbox_results_3.json"
#
# annotation_ensemble = "../ensemble.json"

# image_folder_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/images"
# GT_path = '/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/kvasir_seg_COCO.json'
# annotation1 = "../test_kvasir_bbox_results_0.json"
# annotation2 = "../test_kvasir_bbox_results_1.json"
# annotation3 = "../test_kvasir_bbox_results_2.json"
# annotation4 = "../test_kvasir_bbox_results_3.json"
#
# annotation_ensemble = "../ensemble_kvasir.json"

image_folder_path = "../datasets/polyps_c5/test"
GT_path = '../datasets/polyps_c5/annotations/instances_test.json'
annotation1 = "../test_kvasir_model_on_endocv_center_bbox_results_0.json"
annotation2 = "../test_kvasir_model_on_endocv_center_bbox_results_1.json"
annotation3 = "../test_kvasir_model_on_endocv_center_bbox_results_2.json"
annotation4 = "../test_kvasir_model_on_endocv_center_bbox_results_3.json"

annotation_ensemble = "../ensemble_kvasir_on_endocv_center.json"

f = open(GT_path)
GT_annotations = json.load(f)

f = open(annotation1)
annotations1 = json.load(f)

f = open(annotation2)
annotations2 = json.load(f)

f = open(annotation3)
annotations3 = json.load(f)

f = open(annotation4)
annotations4 = json.load(f)

f = open(annotation_ensemble)
annotations_ensemble = json.load(f)

colors_for_classes = ["yellow", "cyan", "magenta", "darkorange", "lime"]


def draw_bbox_on_image_individual(image, annotations1, annotations2, annotations3, annotations4):
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

    for object in annotations4:
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


def draw_bbox_on_image_ensemble(image, annotations_ensemble):
    image = image[:, :, ::-1]
    fig, ax = plt.subplots()
    ax.imshow(image)

    for object in annotations_ensemble:
        rect = patches.Rectangle((object["bbox"][0], object["bbox"][1]),
                                 object["bbox"][2],
                                 object["bbox"][3],
                                 linewidth=2,
                                 edgecolor=colors_for_classes[4],
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(object["bbox"][0],
                 object["bbox"][1] - 3,
                 "{:.2f}".format(object["score"]),
                 color=colors_for_classes[4])

    plt.tight_layout()
    plt.axis("off")
    plt.show()


image_id = random.randint(0, 200)
# image_id = 32
image_name = GT_annotations["images"][image_id]["file_name"]

image_annotations1 = [x for x in annotations1 if x["image_id"] == image_id]
image_annotations2 = [x for x in annotations2 if x["image_id"] == image_id]
image_annotations3 = [x for x in annotations3 if x["image_id"] == image_id]
image_annotations4 = [x for x in annotations4 if x["image_id"] == image_id]

image_annotations_ensemble = [x for x in annotations_ensemble if x["image_id"] == image_id]

image = cv2.imread(os.path.join(image_folder_path, image_name))
print(str(image_id) + " " + image_name)
draw_bbox_on_image_individual(image, image_annotations1, image_annotations2, image_annotations3, image_annotations4)
draw_bbox_on_image_ensemble(image, image_annotations_ensemble)
