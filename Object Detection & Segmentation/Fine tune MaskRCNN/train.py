from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchvision
import torch
from torch.utils.data import DataLoader
import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import math
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Train MaskRCNN on a single class from MS COCO
CAT_ID = 58  # Object category ID from MS COCO, 58=hot dog


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    return model


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_id, coco):
        self.img_id = img_id
        self.coco = coco
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        img = self.coco.loadImgs(self.img_id[index])[0]
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=44)
        anns = self.coco.loadAnns(annIds)
        img_file = img['file_name']
        image = Image.open(f'coco/train2017/{img_file}').convert('RGB')
        image = self.transform_img(image)
        masks = []
        for i in range(len(anns)):
            mask = np.zeros((img['height'], img['width']))
            mask = np.maximum(self.coco.annToMask(anns[i]), mask)
            masks.append(mask)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = []
        for i in range(len(anns)):
            pos = anns[i]['bbox']
            xmin = pos[0]
            xmax = xmin + pos[2]
            ymin = pos[1]
            ymax = ymin + pos[3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(anns),), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return image, target


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    mse_loss_total = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images, targets)
        masks = outputs[0]['masks']
        target_masks = targets[0]['masks']
        masks_len = min(len(masks), len(target_masks))
        if masks_len == 0:
            continue
        masks = masks[:masks_len]
        target_masks = masks[:masks_len]
        mse = torch.nn.functional.mse_loss(masks, target_masks)
        mse_loss_total += mse

    print(f'*******Evaluation MSE Loss : {mse_loss_total} ********')
    metric_logger.update(mse_loss=mse_loss_total)

    return metric_logger


def train():
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    annFile = 'coco/annotations_trainval2017/annotations/instances_train2017.json'
    coco = COCO(annFile)
    img_Ids = coco.getImgIds(catIds=CAT_ID)
    train_imgIds, val_imgIds = train_test_split(img_Ids, test_size=0.1)
    train_dataset = Dataset(train_imgIds, coco)
    data_loader = DataLoader(
        train_dataset, batch_size=4, num_workers=os.cpu_count(),
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        torch.save(model.state_dict(), "model.pth")


if __name__ == '__main__':
    train()
