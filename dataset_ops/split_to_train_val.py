# Created by Gorkem Polat at 9.02.2021
# contact: polatgorkem@gmail.com

import os
import glob
import shutil
from random import shuffle

root_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper"
val_ratio = 0.176

train_path = os.path.join(root_path, "train")
val_path = os.path.join(root_path, "val")

if os.path.isdir(train_path):
    shutil.rmtree(train_path)
os.mkdir(train_path)
if os.path.isdir(val_path):
    shutil.rmtree(val_path)
os.mkdir(val_path)

all_files = glob.glob(os.path.join(root_path, "train_val", "*.jpg"))
all_txt_files = glob.glob(os.path.join(root_path, "train_val", "*.txt"))
shuffle(all_files)

val_files = all_files[:int(len(all_files) * val_ratio)]
train_files = all_files[int(len(all_files) * val_ratio):]

for train_file_path in train_files:
    file_name = train_file_path.split("/")[-1][:-4]
    shutil.copyfile(train_file_path, os.path.join(train_path, file_name + ".jpg"))

    if train_file_path[:-4] + "_mask.txt" in all_txt_files:
        shutil.copyfile(train_file_path[:-4] + "_mask.txt", os.path.join(train_path, file_name + ".txt"))
    elif train_file_path[:-4] + ".txt" in all_txt_files:
        shutil.copyfile(train_file_path[:-4] + ".txt", os.path.join(train_path, file_name + ".txt"))
    else:
        raise Exception("Error on txt name: " + train_file_path)

for val_file_path in val_files:
    file_name = val_file_path.split("/")[-1][:-4]
    shutil.copyfile(val_file_path, os.path.join(val_path, file_name + ".jpg"))

    if val_file_path[:-4] + "_mask.txt" in all_txt_files:
        shutil.copyfile(val_file_path[:-4] + "_mask.txt", os.path.join(val_path, file_name + ".txt"))
    elif val_file_path[:-4] + ".txt" in all_txt_files:
        shutil.copyfile(val_file_path[:-4] + ".txt", os.path.join(val_path, file_name + ".txt"))
    else:
        raise Exception("Error on txt name: " + val_file_path)
