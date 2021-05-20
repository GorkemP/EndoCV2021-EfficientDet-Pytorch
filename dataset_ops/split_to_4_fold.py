# Created by Gorkem Polat at 6.03.2021
# contact: polatgorkem@gmail.com

import os
import glob
import random
import shutil
import numpy as np

# fix seed in order to get same folds for reproducibility
random.seed(35)
np.random.seed(35)

root_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/paper"
fold_num = 4
fold_folder_prefix = "fold"

folds_path = os.path.join(root_path, "CV_folds")
if os.path.isdir(folds_path):
    shutil.rmtree(folds_path)
os.mkdir(folds_path)

all_files = glob.glob(os.path.join(root_path, "train_val", "*.jpg"))
all_txt_files = glob.glob(os.path.join(root_path, "train_val", "*.txt"))
random.shuffle(all_files)

files_splitted = np.array_split(all_files, fold_num)

for fold in range(fold_num):
    fold_path = os.path.join(folds_path, fold_folder_prefix + "_" + str(fold))
    os.mkdir(fold_path)

    fold_train_path = os.path.join(fold_path, "train")
    os.mkdir(fold_train_path)

    fold_val_path = os.path.join(fold_path, "val")
    os.mkdir(fold_val_path)

    train_files = []
    for i in range(fold_num):
        if fold != i:
            train_files.extend(files_splitted[i])

    val_files = files_splitted[fold]

    for train_file_path in train_files:
        file_name = train_file_path.split("/")[-1][:-4]
        shutil.copyfile(train_file_path, os.path.join(fold_train_path, file_name + ".jpg"))

        if train_file_path[:-4] + "_mask.txt" in all_txt_files:
            shutil.copyfile(train_file_path[:-4] + "_mask.txt", os.path.join(fold_train_path, file_name + ".txt"))
        elif train_file_path[:-4] + ".txt" in all_txt_files:
            shutil.copyfile(train_file_path[:-4] + ".txt", os.path.join(fold_train_path, file_name + ".txt"))
        else:
            raise Exception("Error on txt name: " + train_file_path)

    for val_file_path in val_files:
        file_name = val_file_path.split("/")[-1][:-4]
        shutil.copyfile(val_file_path, os.path.join(fold_val_path, file_name + ".jpg"))

        if val_file_path[:-4] + "_mask.txt" in all_txt_files:
            shutil.copyfile(val_file_path[:-4] + "_mask.txt", os.path.join(fold_val_path, file_name + ".txt"))
        elif val_file_path[:-4] + ".txt" in all_txt_files:
            shutil.copyfile(val_file_path[:-4] + ".txt", os.path.join(fold_val_path, file_name + ".txt"))
        else:
            raise Exception("Error on txt name: " + val_file_path)
