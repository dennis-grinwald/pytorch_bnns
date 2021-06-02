import argparse
import os
import sys
import random
import json
import shutil
import time
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pbnn.datasets import get_datasets
from pbnn.models.mcdo_resnets import mcdo_resnet18, mcdo_resnet50

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Load the data
valdir = '/home/dgrinwald/datasets/places365/val/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True
)

# Load the model and criterion
model_ckpt_path = '/home/dgrinwald/projects/pytorch_bnns/models/places365/mcdo_resnet18_bs_256_lr_0.1_dropout_0.05_wd_0.0001_mo_0.9_dist_training/model_best_1.pth.tar'
model_ckpt = torch.load(model_ckpt_path)

# REMOVE THE MODULE PREFIX #
new_state_dict = OrderedDict()
for k, v in model_ckpt['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model = mcdo_resnet18(p=0.05, num_classes=365)
model.load_state_dict(new_state_dict)
#print(model)

# Validation loop

# switch to evaluate mode
device = torch.device('cuda:0')

model.to(device).eval()

with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)    
        target = target.to(device)
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        print(f'Acc1: {acc1}, Acc5: {acc5}')

