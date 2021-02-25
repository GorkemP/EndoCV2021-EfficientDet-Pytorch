# Created by Gorkem Polat at 25.02.2021
# contact: polatgorkem@gmail.com
# original author: signatrix, zylo117
# adapted from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
# modified by GorkemP

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
from utils.augmentations import CustomAugmenter

import wandb
from statistics import mean

# COCO imports
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

enable_wandb = True

project_name = "polyps"
efficientdet_version = 0
num_worker = 4
batch_size = 10
lr = 0.01
num_epochs = 100
head_only = False
weights_file = "weights/efficientdet-d" + str(efficientdet_version) + ".pth"
early_stopping_patience = 12
lr_scheduler_patience = 5
mAP_interval = 5

if enable_wandb:
    wandb.init(project="endocv2021", entity="gorkemp", save_code=True)
    wandb.run.name = os.path.basename(__file__)[:-3] + "_" + wandb.run.name.split("-")[2]
    wandb.run.save()

    config = wandb.config
    config.project_name = project_name
    config.configuration = efficientdet_version
    config.num_worker = num_worker
    config.batch_size = batch_size
    config.lr = lr
    config.num_epochs = num_epochs
    config.num_worker = num_worker
    config.early_stopping_patience = early_stopping_patience

# COCO variables
compound_coef = efficientdet_version
nms_threshold = 0.5
use_cuda = True
gpu = 0
use_float16 = False
weights_path = f"logs/{project_name}/efficientdet-d{efficientdet_version}_best.pth"

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[efficientdet_version],
                                                         mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id'   : image_id,
                    'category_id': label + 1,
                    'score'      : float(score),
                    'bbox'       : box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


def get_mAP(opt, set_type="val_set"):
    """
    return AP@[0.50:0.95] and AP@0.50
    """
    SET_NAME = params[set_type]
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    model = EfficientDetBackbone(compound_coef=efficientdet_version, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(gpu)

        if use_float16:
            model.half()

    evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    COCO_result = _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
    return COCO_result[0], COCO_result[1]


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


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(opt):
    early_stop_counter = 0

    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size' : opt.batch_size,
                       'shuffle'    : True,
                       'drop_last'  : True,
                       'collate_fn' : collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size' : opt.batch_size,
                  'shuffle'    : False,
                  'drop_last'  : True,
                  'collate_fn' : collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             CustomAugmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    last_step = 0
    print('[Info] initializing weights...')
    init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=lr_scheduler_patience,
                                                           verbose=True)

    epoch = 0
    best_loss = 1e5
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            model.train()
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            # Training Epoch
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    train_cls_loss, train_reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    train_cls_loss = train_cls_loss.mean()
                    train_reg_loss = train_reg_loss.mean()

                    train_loss = train_cls_loss + train_reg_loss
                    if train_loss == 0 or not torch.isfinite(train_loss):
                        continue

                    train_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(train_loss))

                    progress_bar.set_description(
                            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                                    step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, train_cls_loss.item(),
                                    train_reg_loss.item(), train_loss.item()))
                    step += 1

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            epoch_mean_train_loss = mean(epoch_loss)

            # Validation

            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(val_generator):
                with torch.no_grad():
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    val_cls_loss, val_reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    val_cls_loss = val_cls_loss.mean()
                    val_reg_loss = val_reg_loss.mean()

                    val_loss = val_cls_loss + val_reg_loss
                    if val_loss == 0 or not torch.isfinite(val_loss):
                        continue

                    loss_classification_ls.append(val_cls_loss.item())
                    loss_regression_ls.append(val_reg_loss.item())

            val_cls_loss = np.mean(loss_classification_ls)
            val_reg_loss = np.mean(loss_regression_ls)
            val_loss = val_cls_loss + val_reg_loss
            scheduler.step(val_loss)

            print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                            epoch, opt.num_epochs, val_cls_loss, val_reg_loss, val_loss))
            if val_loss + opt.es_min_delta < best_loss:
                early_stop_counter = 0
                best_loss = val_loss
                print("overwriting the best model!")
                if enable_wandb:
                    wandb.run.summary["best loss"] = best_loss
                save_checkpoint(model, f'efficientdet-d{opt.compound_coef}' + '_best.pth')
            else:
                early_stop_counter += 1

            if ((epoch + 1) % mAP_interval == 0) and (epoch > 0):
                if enable_wandb:
                    train_mAP_COCO, train_mAP_50 = get_mAP(opt, "train_set")
                    val_mAP_COCO, val_mAP_50 = get_mAP(opt, "val_set")
                    wandb.log({"epoch"          : epoch,
                               "train mAP COCO" : train_mAP_COCO,
                               "train mAP @0.50": train_mAP_50,
                               "val mAP COCO"   : val_mAP_COCO,
                               "val mAP @0.50"  : val_mAP_50})

            if enable_wandb:
                wandb.log(
                        {"epoch"     : epoch,
                         "lr"        : optimizer.param_groups[0]['lr'],
                         'train loss': epoch_mean_train_loss,
                         'val Loss'  : val_loss})

            if early_stop_counter >= opt.es_patience:
                print("Early stopping at: " + str(epoch))
                break

    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        if enable_wandb:
            wandb.run.finish()

    if enable_wandb:
        wandb.run.finish()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
