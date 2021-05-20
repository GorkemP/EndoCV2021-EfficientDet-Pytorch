# Created by Gorkem Polat at 15.03.2021
# contact: polatgorkem@gmail.com

import os
import glob
import shutil
import random

all_images_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/images"
train_folder = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/train"
test_folder = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper/Kvasir-SEG/test"
test_ratio = 0.2

if os.path.isdir(train_folder):
    shutil.rmtree(train_folder)
os.mkdir(train_folder)

if os.path.isdir(test_folder):
    shutil.rmtree(test_folder)
os.mkdir(test_folder)

images = glob.glob(os.path.join(all_images_path, "*.jpg"))
random.shuffle(images)

test_images = images[:int(len(images) * test_ratio)]
train_images = images[int(len(images) * test_ratio):]

for test_image in test_images:
    file_name = test_image.split("/")[-1]
    shutil.copyfile(test_image, os.path.join(test_folder, file_name))

for train_image in train_images:
    file_name = train_image.split("/")[-1]
    shutil.copyfile(train_image, os.path.join(train_folder, file_name))
