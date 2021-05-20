import os
import glob
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm


def resize_by_keeping_ratio(image, new_height, fixed_width):
    height, width, _ = image.shape
    scale = new_height / height
    new_width = int(scale * width)
    if new_width > fixed_width:
        print("big image: " + str(new_width))
        new_width = fixed_width

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    new_image = np.zeros((new_height, fixed_width, 3))
    new_image[0:new_height, 0:new_width] = resized_image

    return new_image, new_height, fixed_width, scale


def resize_by_keeping_ratio_square(image, new_width, fixed_height):
    height, width, _ = image.shape
    scale = new_width / width
    new_height = int(scale * height)
    if new_height > fixed_height:
        print("outlier image, height>width: " + str(new_height))
        new_height = fixed_height

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    new_image = np.zeros((fixed_height, new_width, 3))
    new_image[0:new_height, 0:new_width] = resized_image

    return new_image, fixed_height, new_width, scale


def resize_by_width(image, new_width):
    height, width, _ = image.shape
    scale = new_width / width
    new_height = int(scale * height)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image, new_height, new_width, scale


# For CV folds
fold_id = 3
target_root_name = os.path.join("../datasets/polyps_paper_" + str(fold_id))
target_set_name = "val"
source_files_path = os.path.join("/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper", "CV_folds",
                                 "fold_" + str(fold_id), target_set_name)

print("fold: " + str(fold_id) + " | " + " target: " + target_set_name)


# For single split
# target_root_name = os.path.join("../datasets/polyps_paper")
# target_set_name = "test"
# source_files_path = os.path.join("/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper", target_set_name)

classes = ["polyp"]
new_image_width = 1024

os.makedirs(target_root_name, exist_ok=True)
os.makedirs(os.path.join(target_root_name, "annotations"), exist_ok=True)

if os.path.isdir(os.path.join(target_root_name, target_set_name)):
    shutil.rmtree(os.path.join(target_root_name, target_set_name))
if os.path.isfile(os.path.join(target_root_name, "annotations", "instances_" + target_set_name + ".json")):
    os.remove(os.path.join(target_root_name, "annotations", "instances_" + target_set_name + ".json"))
os.makedirs(os.path.join(target_root_name, target_set_name))

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

image_paths = glob.glob(os.path.join(source_files_path, "*.jpg"))

image_counter = 0
annotation_counter = 0
for image_path in tqdm(image_paths):
    image_name = image_path.split("/")[-1][:-4]

    with open(image_path[:-4] + ".txt") as f:
        bbox_annotations = f.readlines()

    # eliminate duplicate lines
    bbox_annotations = list(set(bbox_annotations))
    try:
        if len(bbox_annotations) > 0 and not ("\n" in bbox_annotations):
            image = cv2.imread(image_path)
            # image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            new_image, new_height, new_width, scale = resize_by_width(image, new_image_width)

            for bbox_annotation in bbox_annotations:
                bbox_info = bbox_annotation.split(" ")

                category_id = classes.index(bbox_info[0]) + 1

                x = int(int(bbox_info[1]) * scale)
                y = int(int(bbox_info[2]) * scale)
                object_width = int((int(bbox_info[3]) - int(bbox_info[1])) * scale)
                object_height = int((int(bbox_info[4]) - int(bbox_info[2])) * scale)

                annotation_dict = {}
                annotation_dict["id"] = annotation_counter
                annotation_dict["image_id"] = image_counter
                annotation_dict["category_id"] = category_id
                annotation_dict["iscrowd"] = 0
                annotation_dict["area"] = object_width * object_height
                annotation_dict["bbox"] = [x, y, object_width, object_height]
                annotations["annotations"].append(annotation_dict)
                annotation_counter += 1

            cv2.imwrite(os.path.join(target_root_name, target_set_name, str(image_counter) + ".jpg"), new_image)

            image_dict = {}
            image_dict["id"] = image_counter
            image_dict["file_name"] = str(image_counter) + ".jpg"
            image_dict["original_file_name"] = image_name + ".jpg"
            image_dict["width"] = new_width
            image_dict["height"] = new_height
            annotations["images"].append(image_dict)
            image_counter += 1
        else:
            print("No bbox information for: " + image_name)
    except Exception as e:
        print("image path:" + image_path)
        print(str(e.message))
        break

with open(os.path.join(target_root_name, "annotations", "instances_" + target_set_name + ".json"), "w") as outfile:
    json.dump(annotations, outfile)
