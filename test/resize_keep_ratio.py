# Created by Gorkem Polat at 10.02.2021
# contact: polatgorkem@gmail.com

import os
import glob
import json
import shutil
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def show_image(image):
    plt.imshow(image)
    plt.show()


def show_image_opencv(image):
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap="gray")
    plt.show()


def resize_by_keeping_ratio(image, new_height, fixed_width):
    height, width, _ = image.shape
    scale = new_height / height
    new_width = int(scale * width)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    new_image = np.zeros((new_height, fixed_width, 3))
    new_image[0:new_height, 0:new_width] = resized_image

    return new_image, scale


image_path = "/home/gorkem/Desktop/data/EndoCV2021/original_files/trainData_EndoCV2021_5_Feb2021/data_C1/bbox_image"
image_paths = glob.glob(os.path.join(image_path, "*.jpg"))
image_path = random.choice(image_paths)
image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

new_image, scale = resize_by_keeping_ratio(image, 512, 910)
print(image_path)
print("height: " + str(image.shape[0]) + " width: " + str(image.shape[1]))

new_image = new_image / 255
# show_image(new_image)
show_image_opencv(new_image.astype("float32"))
