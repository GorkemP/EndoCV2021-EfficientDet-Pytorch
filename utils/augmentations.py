# Created by Gorkem Polat at 23.02.2021
# contact: polatgorkem@gmail.com

import cv2
import numpy as np
import albumentations as A


class CustomAugmenter(object):
    """Apply augmentations"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image.astype(np.float32)
        transformed = self.transform(image=image, bboxes=annots)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        # if no bbox, return empty 2D numpy array
        if len(transformed_bboxes) != 0:
            transformed_bboxes_np = np.array([np.array(xi) for xi in transformed_bboxes])
        else:
            transformed_bboxes_np = np.zeros((0, 5))

        sample = {'img': transformed_image, 'annot': transformed_bboxes_np}

        return sample


class Resize_by_keeping_dimensions(object):
    def __init__(self, scale):
        self.scale = scale
        self.scale_transform = A.Compose([
            scale
        ], bbox_params=A.BboxParams(format="pascal_voc",
                                    min_visibility=0.25))

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image.astype(np.float32)
        height, width, _ = image.shape

        transformed = self.scale_transform(image=image, bboxes=annots)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        new_height, new_width, _ = transformed_image.shape

        if new_width > width:
            crop_transform = A.Compose([
                A.RandomCrop
            ], bbox_params=A.BboxParams(format="pascal_voc",
                                        min_visibility=0.25))
            # TODO incomplete implementation
            pass
        else:
            new_image = np.zeros((height, width, 3))
            new_image[0:new_height, 0:new_width] = transformed_image

        # if no bbox, return empty 2D numpy array
        if len(transformed_bboxes) != 0:
            transformed_bboxes_np = np.array([np.array(xi) for xi in transformed_bboxes])
        else:
            transformed_bboxes_np = np.zeros((0, 5))

        sample = {'img': transformed_image, 'annot': transformed_bboxes_np}

        return sample


class CustomAugmenter_experimental(object):
    """Apply augmentations"""

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)
            # A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.25))

        transformed = transform(image=image, bboxes=annots)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        # if no bbox, return empty 2D numpy array
        if len(transformed_bboxes) != 0:
            transformed_bboxes_np = np.array([np.array(xi) for xi in transformed_bboxes])
        else:
            transformed_bboxes_np = np.zeros((0, 5))

        sample = {'img': transformed_image, 'annot': transformed_bboxes_np}

        return sample
