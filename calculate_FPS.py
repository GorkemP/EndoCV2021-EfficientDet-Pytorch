# Created by Gorkem Polat at 12.03.2021
# contact: polatgorkem@gmail.com

import torch
from torch.backends import cudnn
import glob
import yaml
import time

from backbone import EfficientDetBackbone
import cv2

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

project_name = "polyps_paper"
compound_coef = 2
force_input_size = None  # set None to use default size

images_full_path = "/home/ws2080/Desktop/data/EndoCV2021/edited_files/val/*.jpg"
img_paths = glob.glob(images_full_path)

weight_file = f"trained_weights/efficientdet-d2_best_108.pth"

threshold = 0.3
iou_threshold = 0.4  # Used for NMS

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef,
                             num_classes=len(obj_list),
                             ratios=eval(params['anchors_ratios']),
                             scales=eval(params['anchors_scales']))

model.load_state_dict(torch.load(weight_file))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

program_start = time.time()
total_elapsed = 0
for img_path in img_paths:
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    start = time.time()

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    end = time.time()

    out = invert_affine(framed_metas, out)

    total_elapsed += (end - start)

program_end = time.time()
total_time = program_end - program_start

print("total time: " + str(total_time))
print("total processing: " + str(total_elapsed))
print("FPS: " + str(len(img_paths) / total_elapsed))
