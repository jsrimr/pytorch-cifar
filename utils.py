'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import copy
import os
import sys
import time
from collections import OrderedDict

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models import MobileNetV2

SMALL_NOISE = 1e-3 + 1


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# Training
def train(net, criterion, optimizer, trainloader, epoch, device, run):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        run["train/loss"].log(loss.item())
        run["train/acc"].log(100. * correct / total)
    print(f"{epoch}epoch train loss = {loss.item()}")
    print(f"{epoch}epoch train acc = {100. * correct / total}")


best_acc = 0.
def test(net, optimizer, scheduler, criterion, testloader, epoch, device, run, exp_name):
    global best_acc
    net.eval()
    best_acc = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    print(f"EVAL accuracy = {acc}")
    run["eval/acc"].log(acc)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{exp_name}_ckpt.pth')
        best_acc = acc


from models.mobilenetv2 import Block


def net_transform_wider_update(parent_net: MobileNetV2, child_net: MobileNetV2, stage_idx):
    # 기본적으로 같은 거 복사
    new_state_dict = OrderedDict()
    differnt_keys = set()
    for (parent_k, parent_v), (child_k, child_v) in zip(parent_net.state_dict().items(),
                                                        child_net.state_dict().items()):
        if parent_v.shape == child_v.shape:
            new_state_dict[child_k] = parent_v
        else:
            differnt_keys.add(child_k)
            new_state_dict[child_k] = child_v
    child_net.load_state_dict(new_state_dict)

    parent_block_list = list(list(parent_net.layers)[stage_idx])
    child_block_list = list(list(child_net.layers)[stage_idx])
    # 차이나는 부분 핸들
    block_weight_list = []
    for i, (parent_block1, child_block1) in enumerate(zip(parent_block_list, child_block_list)):
        is_last_stage = (stage_idx == len(child_net.layers) - 1) or (stage_idx == -1)
        if is_last_stage:
            parent_block2 = parent_net.conv2
            child_block2 = child_net.conv2
        else:
            if (i + 1) < len(parent_block_list):
                parent_block2 = parent_block_list[i + 1]
                child_block2 = child_block_list[i + 1]
            else:
                parent_block2 = list(list(parent_net.layers)[stage_idx + 1])[0]
                child_block2 = list(list(child_net.layers)[stage_idx + 1])[0]

        if block_weight_list:  # parent_block1 으로 c_in 한번 변환된거 사용함.
            tmp_block = Block(child_block1.in_planes, parent_block1.out_planes, parent_block1.expansion,
                              parent_block1.stride)
            tmp_block.load_state_dict(block_weight_list[-1], strict=False)

            new_weight1, new_weight2 = block_update(tmp_block, parent_block2, child_block1, child_block2, is_last_stage)
            block_weight_list[-1] = new_weight1
            block_weight_list.append(new_weight2)
        else:
            block_weight_list.extend(
                block_update(parent_block1, parent_block2, child_block1, child_block2, is_last_stage))

    for weight, child_block in zip(block_weight_list, child_block_list):
        try:
            child_block.load_state_dict(weight)
        except:
            pass
    # 다음 스테이지의 첫번째 블록 conv1 레이어 업데이트
    child_block2.load_state_dict(block_weight_list[-1])


# train from net_transform
BASE_CFG = [[1, 16, 1, 1],
            [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 16, 2, 2],
            [6, 32, 3, 2],
            [6, 48, 2, 1],
            [6, 96, 2, 2],
            [6, 320, 1, 1]]

TARGET_CFG = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),
    (6, 32, 3, 2),  # 16->32 , 2->3
    (6, 64, 4, 2),  # 32->64 , 3->4
    (6, 96, 3, 1),  # 48->96 , 2->3
    (6, 160, 3, 2),  # 96->160, 2->3
    (6, 380, 1, 1)]


def get_next_net(current_net, current_cfg, plan, plan_idx):
    target, stage_idx = plan[plan_idx]
    if target == "width":
        new_cfg = copy.deepcopy(current_cfg)
        new_cfg[stage_idx][1] = TARGET_CFG[stage_idx][1]
        next_net = MobileNetV2(cfg=new_cfg)

        net_transform_wider_update(current_net, next_net, stage_idx)

    elif target == "depth":
        new_cfg = copy.deepcopy(current_cfg)
        new_cfg[stage_idx][2] = TARGET_CFG[stage_idx][2]
        next_net = MobileNetV2(cfg=new_cfg)

        next_net.load_state_dict(current_net.state_dict(), strict=False)
        next_net.layers[stage_idx][-1].bn3 = process_bn(next_net.layers[stage_idx][-1].bn3)

    return next_net, new_cfg


from box import Box


class BlockWeight:
    pass

    def __init__(self, state_dict: OrderedDict):
        self.conv1 = Box({k[6:]: v for k, v in state_dict.items() if k.startswith("conv1")})
        self.bn1 = Box({k[4:]: v for k, v in state_dict.items() if k.startswith("bn1")})
        self.conv2 = Box({k[6:]: v for k, v in state_dict.items() if k.startswith("conv2")})
        self.bn2 = Box({k[4:]: v for k, v in state_dict.items() if k.startswith("bn2")})
        self.conv3 = Box({k[6:]: v for k, v in state_dict.items() if k.startswith("conv3")})
        self.bn3 = Box({k[4:]: v for k, v in state_dict.items() if k.startswith("bn3")})
        self.shortcut = Box({k[9:]: v for k, v in state_dict.items() if k.startswith("shortcut")})


def block_update(block1, block2, wider_block1, wider_block2, last_stage=False):
    """

    :param block1_weight:
    :param block2_weight:
    :param wider_block1:
    :param wider_block2:
    :param last_stage:
    :return: state_dict
    """
    # BLOCK1 update
    new_state_dict1 = OrderedDict()

    conv_3_w_wider = get_wider_weight(block1.conv3.weight, wider_block1.conv3.weight, axis=0)
    bn_3_w_wider = process_bn_weight(block1.bn3, wider_block1.bn3)

    if wider_block1.shortcut:
        shortcut_wider = get_wider_weight(block1.shortcut[0].weight, wider_block1.shortcut[0].weight,
                                          axis=0)
        bn_shortcut_wider = process_bn_weight(block1.shortcut[1], wider_block1.shortcut[1])
        new_state_dict1['shortcut.0.weight'] = shortcut_wider
        new_state_dict1['shortcut.1.weight'] = bn_shortcut_wider[:, 0]
        new_state_dict1['shortcut.1.bias'] = bn_shortcut_wider[:, 1]
        new_state_dict1['shortcut.1.running_mean'] = bn_shortcut_wider[:, 2]
        new_state_dict1['shortcut.1.running_var'] = bn_shortcut_wider[:, 3]
        new_state_dict1['shortcut.1.num_batches_tracked'] = block1.shortcut[1].state_dict()['num_batches_tracked']

    if last_stage:
        # block2 가 그냥 conv_layer 임
        wider_block2_conv1_w = get_wider_weight(block2.weight, wider_block2.weight, axis=1)
    else:
        wider_block2_conv1_w = get_wider_weight(block2.conv1.weight, wider_block2.conv1.weight, axis=1)

    new_state_dict1['conv1.weight'] = block1.conv1.state_dict()['weight']
    new_state_dict1['bn1.weight'] = block1.bn1.state_dict()['weight']
    new_state_dict1['bn1.bias'] = block1.bn1.state_dict()['bias']
    new_state_dict1['bn1.running_mean'] = block1.bn1.state_dict()['running_mean']
    new_state_dict1['bn1.running_var'] = block1.bn1.state_dict()['running_var']
    new_state_dict1['bn1.num_batches_tracked'] = block1.bn1.state_dict()['num_batches_tracked']

    new_state_dict1['conv2.weight'] = block1.conv2.state_dict()['weight']
    new_state_dict1['bn2.weight'] = block1.bn2.state_dict()['weight']
    new_state_dict1['bn2.bias'] = block1.bn2.state_dict()['bias']
    new_state_dict1['bn2.running_mean'] = block1.bn2.state_dict()['running_mean']
    new_state_dict1['bn2.running_var'] = block1.bn2.state_dict()['running_var']
    new_state_dict1['bn2.num_batches_tracked'] = block1.bn2.state_dict()['num_batches_tracked']

    new_state_dict1['conv3.weight'] = conv_3_w_wider
    new_state_dict1['bn3.weight'] = bn_3_w_wider[:, 0]
    new_state_dict1['bn3.bias'] = bn_3_w_wider[:, 1]
    new_state_dict1['bn3.running_mean'] = bn_3_w_wider[:, 2]
    new_state_dict1['bn3.running_var'] = bn_3_w_wider[:, 3]
    new_state_dict1['bn3.num_batches_tracked'] = block1.bn3.state_dict()['num_batches_tracked']

    # wider_block1.load_state_dict(new_state_dict)

    # BLOCK2 update
    new_state_dict2 = OrderedDict()
    if last_stage:
        new_state_dict2['weight'] = wider_block2_conv1_w
        # wider_block2.load_state_dict(new_state_dict2)
        return [new_state_dict1, new_state_dict2]
    else:
        # conv_1, bn1 c_out expand
        wider_block2_conv1_w = get_wider_weight(wider_block2_conv1_w,
                                                wider_block2.conv1.weight, 0)
        bn_1_w_wider = process_bn_weight(block2.bn1, wider_block2.bn1)

        # conv_2 update
        wider_block2_conv2_w = get_wider_weight(block2.conv2.weight,
                                                wider_block2.conv2.weight, axis=0)
        # bn2 update
        bn_2_w_wider = process_bn_weight(block2.bn2, wider_block2.bn2)
        # new_weights.append(bn_1_w_wider)

        # conv_3 update with normalization, bn3 는 shape 업데이트 필요없음
        wider_block2_conv3_w = get_wider_weight(block2.conv3.weight,
                                                wider_block2.conv3.weight, axis=1)
        if block2.shortcut:  # 쓰일일이 없을 것 같음
            shortcut_wider = get_wider_weight(block2.shortcut[0].weight,
                                              wider_block2.shortcut[0].weight,
                                              axis=1)
            new_state_dict2['shortcut.0.weight'] = shortcut_wider
            new_state_dict2['shortcut.1.weight'] = block2.shortcut[1].state_dict()['weight']
            new_state_dict2['shortcut.1.bias'] = block2.shortcut[1].state_dict()['bias']
            new_state_dict2['shortcut.1.running_mean'] = block2.shortcut[1].state_dict()['running_mean']
            new_state_dict2['shortcut.1.running_var'] = block2.shortcut[1].state_dict()['running_var']
            new_state_dict2['shortcut.1.num_batches_tracked'] = block2.shortcut[1].state_dict()['num_batches_tracked']

        new_state_dict2['conv1.weight'] = wider_block2_conv1_w
        new_state_dict2['bn1.weight'] = bn_1_w_wider[:, 0]
        new_state_dict2['bn1.bias'] = bn_1_w_wider[:, 1]
        new_state_dict2['bn1.running_mean'] = bn_1_w_wider[:, 2]
        new_state_dict2['bn1.running_var'] = bn_1_w_wider[:, 3]
        new_state_dict2['bn1.num_batches_tracked'] = block2.bn1.state_dict()['num_batches_tracked']

        new_state_dict2['conv2.weight'] = wider_block2_conv2_w
        new_state_dict2['bn2.weight'] = bn_2_w_wider[:, 0]
        new_state_dict2['bn2.bias'] = bn_2_w_wider[:, 1]
        new_state_dict2['bn2.running_mean'] = bn_2_w_wider[:, 2]
        new_state_dict2['bn2.running_var'] = bn_2_w_wider[:, 3]
        new_state_dict2['bn2.num_batches_tracked'] = block2.bn2.state_dict()['num_batches_tracked']

        new_state_dict2['conv3.weight'] = wider_block2_conv3_w
        new_state_dict2['bn3.weight'] = block2.bn3.state_dict()['weight']
        new_state_dict2['bn3.bias'] = block2.bn3.state_dict()['bias']
        new_state_dict2['bn3.running_mean'] = block2.bn3.state_dict()['running_mean']
        new_state_dict2['bn3.running_var'] = block2.bn3.state_dict()['running_var']
        new_state_dict2['bn3.num_batches_tracked'] = block2.bn3.state_dict()['num_batches_tracked']

        # wider_block2.load_state_dict(new_state_dict2)
        return [new_state_dict1, new_state_dict2]


def process_bn_weight(parent_bn_w, child_bn_w):
    idx = torch.arange(child_bn_w.weight.shape[0] - parent_bn_w.weight.shape[0])
    bn_w = torch.stack([parent_bn_w.weight, parent_bn_w.bias,
                        parent_bn_w.running_mean,
                        parent_bn_w.running_var], axis=1)
    bn_wider = torch.cat((bn_w, bn_w[idx, :]), axis=0)
    return bn_wider


N = 3
def get_wider_weight(parent_w, child_w, axis):
    """
    :param parent_w:
    :param child_w:
    :param axis: axis = 0 if c_out expansion, axis=1 if c_in expansion
    :return: larger_weight
    """
    idx = torch.arange(child_w.shape[axis] - parent_w.shape[axis])
    if axis == 0:
        new_weight = parent_w[idx, :, :, :]
        larger_weight = torch.cat((parent_w, new_weight), axis=axis)
    else:
        new_weight = parent_w[:, idx, :, :] / N
        larger_weight = torch.cat((parent_w, new_weight), axis=axis)
        larger_weight[:, idx, :, :] = new_weight * (N - 1) / N

    return larger_weight


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
