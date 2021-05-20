# Created by Gorkem Polat at 23.02.2021
# contact: polatgorkem@gmail.com

# import argparse
# import os
# import numpy as np
# import yaml
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from efficientdet.dataset import CocoDataset, Resizer, Normalizer
# from utils.sync_batchnorm import patch_replication_callback
# from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
# from utils.augmentations import CustomAugmenter

import argparse
import os
import traceback

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
import albumentations as A
import cv2

# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.augmentations import CustomAugmenter, CustomAugmenter_experimental

project_name = "polyps"
efficientdet_version = 0
num_worker = 8
batch_size = 10
lr = 0.01
num_epochs = 100
head_only = False
weights_file = "weights/efficientdet-d" + str(efficientdet_version) + ".pth"
early_stopping_patience = 12
lr_scheduler_patience = 5
mAP_interval = 5


def show_torch_data(img):
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()


def draw_bbox_on_image_augmented_numpy(img, annotations):
    image = img
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i in range(len(annotations)):
        rect = patches.Rectangle((annotations[i][0], annotations[i][1]),
                                 annotations[i][2] - annotations[i][0],
                                 annotations[i][3] - annotations[i][1],
                                 linewidth=2,
                                 edgecolor="yellow",
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(annotations[i][0],
                 annotations[i][1] - 3,
                 "polyp",
                 color="yellow")

    plt.tight_layout()
    # plt.axis("off")
    plt.show()


def draw_bbox_on_image_augmented_torch(image, annotations):
    image = image.numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i in range(len(annotations)):
        rect = patches.Rectangle((annotations[i][0], annotations[i][1]),
                                 annotations[i][2] - annotations[i][0],
                                 annotations[i][3] - annotations[i][1],
                                 linewidth=2,
                                 edgecolor="yellow",
                                 facecolor='none')

        ax.add_patch(rect)
        plt.text(annotations[i][0],
                 annotations[i][1] - 3,
                 "polyp",
                 color="yellow")

    plt.tight_layout()
    # plt.axis("off")
    plt.show()


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('EfficientDet Pytorch: GorkemP')
    parser.add_argument('-p', '--project', type=str, default=project_name, help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=efficientdet_version,
                        help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=num_worker, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=head_only,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=100, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=early_stopping_patience,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=weights_file,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


opt = get_args()
params = Params(f'projects/{opt.project}.yml')

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# Normalizer(mean=params.mean, std=params.std),
# training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set)

training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                           transform=transforms.Compose([
                               # Normalizer(mean=params.mean, std=params.std),
                               CustomAugmenter(
                                       A.Compose([
                                           # A.OneOf([
                                           #     A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.2, hue=0.1,
                                           #                   p=0.5),
                                           #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.1,
                                           #                   p=0.5),
                                           # ], p=0.9),
                                           # A.IAAPerspective(),
                                           # A.ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=(-0.8, 1.0),
                                           #                    border_mode=cv2.BORDER_CONSTANT, p=1),
                                           # A.RandomScale(0.5, p=1)
                                           # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=1),
                                           # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                                           # A.Rotate((180), border_mode=cv2.BORDER_CONSTANT),
                                           # A.HorizontalFlip(),
                                           # A.Cutout(num_holes=8, max_h_size=128, max_w_size=128, fill_value=0, p=1)
                                           # A.VerticalFlip()
                                       ], bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.5))
                               ),

                               Resizer(input_sizes[3])
                           ]))

selected_sample = training_set[4]
image_selected = selected_sample["img"]
annotations_selected = selected_sample["annot"]

print(image_selected.shape)
draw_bbox_on_image_augmented_numpy(image_selected, annotations_selected)
# draw_bbox_on_image_augmented_solo(image_selected, annotations_selected)
