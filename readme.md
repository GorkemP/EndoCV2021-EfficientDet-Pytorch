# EndoCV2021 - Addressing generalisability in polyp detection and segmentation challenge & workshop

### This repository can be used to train EfficientDet models on EndoCV2021 dataset and Kvasir-SEG dataset  

### 🏆 The following paper (implementation of which is included in this repository) got the first rank in the polyp detection sub-challenge 

[Polyp Detection in Colonoscopy Images using Deep Learning and Bootstrap Aggregation](http://ceur-ws.org/Vol-2886/paper9.pdf)

```BibTeX
@inproceedings{polat2021polyp,
  title={Polyp Detection in Colonoscopy Images using Deep Learning and Bootstrap Aggregation.},
  author={Polat, Gorkem and Isik Polat, Ece and Kayabay, Kerem and Temizel, Alptekin},
  booktitle={EndoCV@ ISBI},
  pages={90--100},
  year={2021}
}
```
-----
EfficientDet implementation is directly adapted from the [zylo117 EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) repository.

Several additions are made to the standard repository:
- [wandb](https://wandb.ai/site) integration is provided into code. Every **n**th step, training report the mAP of train and validation sets to the wandb platform.
- Early stopping is added. If there is no increase in the mAP of validation set for the last n epoch, training stops.  
- Learning rate scheduling is added. If there is no increase in the mAP of the validation set for the last k epoch, learning rate is decreased with a factor of m.
- [Albumentations](https://albumentations.ai/) library is used to augment dataset.
- Ensemble code that merges different predictions is added. It uses weighted box fusion technique.
- And several example scripts:
  - Inference on single image
  - Visualizations of augmentations

#### Proposed approach
![flowchart of polyp detection framework](/images/pipeline.png)

#### Prediction of individual models (first row) and their ensemble result (second row)
![polyp detection results](/images/polyp_images_final.png)


## How to run

* First, it is advised to experiment with the [original repository](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) to warmup
* Setup environment using ```environment.yml```.
* After downloading the EndoCV2021 dataset, use ```dataset_ops/split_to_4_fold.py``` to create bootstrap folders. Use ```dataset_ops/convert_*``` scripts if necessary.
  * `split_to_4_fold.py` creates CV folds for bootstrap aggregation. For each fold, different project should be created.
  * Since the original repository accepts COCO format, use `convert_to_coco_format_*.py` files to convert annotations.
* Use `train_custom_augmentation` file to run the codes, set necessary parameters like `project_name`,`efficientdet_version`.
* Use `inference_on_4_model.py` to get results for test set.
* Use `ensemble_4.py` to use weighted-box ensemble results.
* In the visualizations folder there are many scripts to visualize bounding boxes, individual and ensemble results.