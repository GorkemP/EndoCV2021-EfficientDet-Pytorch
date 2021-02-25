# Created by Gorkem Polat at 23.02.2021
# contact: polatgorkem@gmail.com

import cv2
import numpy as np
import albumentations as A


class CustomAugmenter(object):
    """Apply augmentations"""

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        transform = A.Compose([
            A.Rotate(180, border_mode=cv2.BORDER_CONSTANT),
            # A.RandomRotate90(p=1)
            A.HorizontalFlip()
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
