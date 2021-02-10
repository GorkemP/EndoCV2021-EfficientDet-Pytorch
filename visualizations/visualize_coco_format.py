# Created by Gorkem Polat at 10.02.2021
# contact: polatgorkem@gmail.com

import os
import json
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

set_name = "train"
image_folder_path = os.path.join("../datasets/polyp", set_name)
annotations_path = "../datasets/polyp/annotations/instances_"+set_name+".json"