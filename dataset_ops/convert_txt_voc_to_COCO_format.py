# Created by Gorkem Polat at 18.03.2021
# contact: polatgorkem@gmail.com

import os
import glob
import shutil
from random import shuffle

original_folder_name = "all_files"
root_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files"

new_path = os.path.join(root_path, original_folder_name + "_test")

if os.path.isdir(new_path):
    shutil.rmtree(new_path)
os.mkdir(new_path)

all_files = glob.glob(os.path.join(root_path, original_folder_name, "*.jpg"))
all_txt_files = glob.glob(os.path.join(root_path, original_folder_name, "*.txt"))
shuffle(all_files)

for file_path in all_files:
    file_name = file_path.split("/")[-1][:-4]
    shutil.copyfile(file_path, os.path.join(new_path, file_name + ".jpg"))

    if file_path[:-4] + "_mask.txt" in all_txt_files:
        shutil.copyfile(file_path[:-4] + "_mask.txt", os.path.join(new_path, file_name + ".txt"))
    elif file_path[:-4] + ".txt" in all_txt_files:
        shutil.copyfile(file_path[:-4] + ".txt", os.path.join(new_path, file_name + ".txt"))
    else:
        raise Exception("Error on txt name: " + file_path)
