import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from models import *
from utils import progress_bar
from models.mobile import MobileNetV2
from models.mobilenetv2_wider import wider_MobileNetV2
from models.mobilenetv2_deeper import deeper_MobileNetV2
import neptune.new as neptune
import numpy as np
device ='cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
best_acc2 = 0  # best test accuracy
start_epoch2 = 0


ncfg= [(1,  16, 1, 1),
       (6,  30, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  40, 3, 2),
       (6,  80, 4, 2),
       (6, 120, 3, 1),
       (6, 200, 3, 2),
       (6, 320, 1, 1)]

net = MobileNetV2()
wider_net = MobileNetV2(cfg = ncfg)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


wider_net = wider_net.to(device)
if device == 'cuda':
    wider_net = torch.nn.DataParallel(wider_net)
    cudnn.benchmark = True

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt09_startfrom150.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print(best_acc)
print(start_epoch)

checkpoint2 = torch.load('./checkpoint/ckpt10_wide_300epoch.pth')
wider_net.load_state_dict(checkpoint2['net'])
best_acc2 = checkpoint2['acc']
start_epoch2 = checkpoint2['epoch']
print(best_acc2)
print(start_epoch2)


for param_tensor in net.state_dict():
    if (net.state_dict()[param_tensor].size() == wider_net.state_dict()[param_tensor].size()):
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        wgt_dif = wider_net.state_dict()[param_tensor].clone().detach() - net.state_dict()[param_tensor].clone().detach()
        wgt_dif = torch.mul(wgt_dif, wgt_dif)
        dif_sum = torch.sum(wgt_dif)
        dif_sum = dif_sum/torch.numel(net.state_dict()[param_tensor])
        print(dif_sum)
    else:
        if(("conv" in param_tensor) or ("shortcut.0" in param_tensor)):
            print("DIFF :", param_tensor, "\t", net.state_dict()[param_tensor].size())
            s1 = net.state_dict()[param_tensor].size()[0]
            s2 = net.state_dict()[param_tensor].size()[1]
            wgt_dif = wider_net.state_dict()[param_tensor].clone().detach()[:s1, :s2, :, :] - net.state_dict()[param_tensor].clone().detach()
            wgt_dif = torch.mul(wgt_dif, wgt_dif)
            dif_sum = torch.sum(wgt_dif)
            dif_sum = dif_sum/torch.numel(net.state_dict()[param_tensor])
            print(dif_sum)