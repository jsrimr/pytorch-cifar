'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def bn_identity(bn_origin):
    bn = deepcopy(bn_origin)
    bn.weight.data = torch.sqrt(bn_origin.running_var + bn_origin.eps)
    bn.bias.data = bn_origin.running_mean  # torch.zeros_like(bn.bias.data) #bn_origin.running_mean

    return bn


def process_bn(bn_origin):
    bn = deepcopy(bn_origin)
    bn.weight.data = torch.zeros_like(bn.weight)
    bn.bias.data = torch.zeros_like(bn.bias)

    return bn


def get_conv_identity_layer(conv_layer):
    w = conv_layer.weight.data
    mid = w.shape[-1] // 2

    # todo : add noise
    for out in range(w.shape[0]):  # 모든 커널 돌며
        idx = out % w.shape[1]
        tmp = torch.zeros_like(w[out])
        tmp[idx, mid, mid] = 1.
        w[out] = tmp

    return w


from models.mobilenetv2 import Block, MobileNetV2

from collections import OrderedDict


def test_net_level():
    CFG = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 5, 2),  # 4 => 5
        (6, 96, 4, 1),  # 3 => 4
        (6, 192, 3, 2),  # 160 => 192
        (6, 380, 1, 1)
    ]  # 320 => 380
    parent_net = MobileNetV2(cfg=CFG)
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    # parent_net.load_state_dict(checkpoint['net'])

    deeper_net_cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 5, 2),
        (6, 96, 5, 1),  # 4 => 5
        (6, 192, 3, 2),
        (6, 380, 1, 1)
    ]  # 320 => 380

    child_net = MobileNetV2(cfg=deeper_net_cfg)
    child_net.load_state_dict(parent_net.state_dict(), strict=False)
    child_net.layers.out_96_block4.bn3 = process_bn(child_net.layers.out_96_block4.bn3)

    child_net.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = child_net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f"accuracy = {100. * correct / total}")


def test_block_level():
    block1 = Block(3, 16, 6, 1)

    exp_block1 = Block(3, 16, 6, 1)
    exp_block2 = Block(16, 16, 6, 1)
    exp_block1.load_state_dict(block1.state_dict())

    x = torch.rand(1, 3, 28, 28)

    res1 = block1(x)
    res2 = exp_block1(x)
    res2 = exp_block2(res2)

    print(f"Before res1==res2 : {torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)}")

    # exp_block2 update
    # conv1_w_identity = get_conv_identity_layer(exp_block2.conv1)
    #
    # conv2_w_identity = get_conv_identity_layer(exp_block2.conv2)
    #
    # conv3_w_identity = get_conv_identity_layer(exp_block2.conv3)

    # exp_block2.conv1.weight.data = conv1_w_identity
    # exp_block2.conv2.weight.data = conv2_w_identity
    # exp_block2.bn2 = bn2_identity
    #
    # exp_block2.conv3.weight.data = conv3_w_identity
    # exp_block2.bn3 = bn3_identity
    # exp_block2.conv1(test_input)[:, :test_input.shape[1], :, :] == test_input

    # test_input = torch.rand(1, 96, 32, 32)
    #
    # # test_input = torch.rand(1, 3, 2, 2)
    # test_bn = nn.BatchNorm2d(3)
    # test_bn(test_input)
    #
    # test_bn.running_mean * test_bn.momentum + test_input.mean(dim=[0, 2, 3]) * (1 - test_bn.momentum)
    # test_bn(test_input)
    # test_bn.running_mean
    #
    # exp_block2.bn1(test_input) == test_input
    #
    # exp_block2.train()
    # exp_block2.bn1 = bn_identity(exp_block2.bn1)
    # exp_block2.bn1.running_mean == exp_block2.bn1.bias
    # exp_block2.bn1.weight == torch.sqrt(exp_block2.bn1.running_var + exp_block2.bn1.eps)
    # exp_block2.eval()
    # exp_block2.bn1(test_input) == test_input
    #
    # exp_block2.bn1(exp_block2.conv1(test_input))[:, :test_input.shape[1], :, :] == test_input

    bn3_identity = process_bn(exp_block2.bn3)
    exp_block2.bn3 = bn3_identity

    res1 = block1(x)

    res2 = exp_block1(x)
    res2 = exp_block2(res2)

    print(f"After res1==res2 : {torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)}")
    assert torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)


def test_layer_level():
    pass


if __name__ == '__main__':
    # test_layer_level()
    # test_block_level()
    test_net_level()
