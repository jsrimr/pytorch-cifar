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
import neptune.new as neptune
import numpy as np

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 

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

ncfg= [(1,  16, 1, 1),
       (6,  30, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  40, 3, 2),
       (6,  80, 4, 2),
       (6, 120, 3, 1),
       (6, 200, 3, 2),
       (6, 320, 1, 1)]

net = MobileNetV2()
wider_net = MobileNetV2(cfg = ncfg)
"""
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



wider_net = wider_net.to(device)
if device == 'cuda':
    wider_net = torch.nn.DataParallel(wider_net)
    cudnn.benchmark = True
"""
def net_transform_wider(net, wider):
    cop_bef = torch.Tensor().to(device)
    bef = ""
    cop_sc = torch.Tensor().to(device)
    bef_sc = ""
    sc = 0
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        if (net.state_dict()[param_tensor].size() == wider.state_dict()[param_tensor].size()):
            weight1 = net.state_dict()[param_tensor]
            wider.state_dict()[param_tensor].copy_(weight1)
            
        else :
            print(param_tensor, "\t", wider.state_dict()[param_tensor].size())
            # Conv3 --> save for shortcut
            if("conv" in param_tensor):
                ## First conv3 block
                diff1 = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                diff2 = wider.state_dict()[param_tensor].shape[1] - net.state_dict()[param_tensor].shape[1]
                cop = (net.state_dict()[param_tensor]).clone().detach()
                mn = torch.mean(cop)
                vr = torch.std(cop)
                if(diff2 < 1):
                    cop_bef = (net.state_dict()[param_tensor]).clone().detach()
                    sh = list(net.state_dict()[param_tensor].shape)
                    sh[0] = diff1
                    new1 = torch.zeros(sh)
                    cop_bef= torch.cat((cop_bef, new1), axis=0)
                    bef = param_tensor[:]

                else :
                    ## 1. Add an extra channel
                    sh2 = list(wider.state_dict()[param_tensor].shape)
                    sh2[1] = diff2
                    sh2[0] -= diff1
                    new2 = torch.normal(mn, vr, size = sh2)
                    cc = torch.cat((cop, new2), axis = 1)
                    #wider.state_dict()[param_tensor].copy_(cc)


                    ## 2. Assign previous channel
                    sz = net.state_dict()[bef].shape[0]
                    sz2 = net.state_dict()[param_tensor].shape[0]
                    c = new2[:sz2].squeeze()
                    print(c.shape)
                    u, s, vh = torch.Tensor(np.linalg.svd(c.numpy(), full_matrices = True, compute_uv=True)).to(device)
                    x = vh[-1, :]
                    for j in range(0, diff2):
                        cop_bef[sz+j:, : , :, :] = x + 0.001 * torch.normal(0, 0.2, size = x.shape)
                    wider.state_dict()[bef].copy_(cop_bef)

                    ## 5. Handle shortcut
                    sz3 = net.state_dict()[bef_sc].shape[0]
                    if(("conv1" in param_tensor) and (sc == 1)):
                        for j in range(0, diff2):
                            cop_sc[sz3+j:, : , :, :] = x + 0.001 * torch.normal(0, 0.2, size = x.shape)
                            wider.state_dict()[bef].copy_(cop_bef)
                        sc = 0

                    ## 4. Update variables

                    ## 3. Add more channels
                    if (diff1 > 0):
                        cop_bef = cc.clone().detach()
                        sh = list(wider.state_dict()[param_tensor].shape)
                        sh[0] = diff1
                        new1 = torch.zeros(sh)
                        cop_bef= torch.cat((cop_bef, new1), axis=0)
                        bef = param_tensor[:]
                        wider.state_dict()[param_tensor].copy_(cc)
                    

            elif("shortcut.0" in param_tensor):
                ## change variable
                sc = 1
                cop_sc = (net.state_dict()[param_tensor]).clone().detach()
                sh = list(net.state_dict()[param_tensor].shape)
                sh[0] = diff1
                new1 = torch.zeros(sh)
                cop_sc= torch.cat((cop_sc, new1), axis=0)
                bef_sc = param_tensor[:]
            
            else:
                cop3 = (net.state_dict()[param_tensor]).clone().detach()
                diff3 = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                new3 = cop3.data[:diff3]
                toadd3 = torch.cat((cop3, new3), axis=0)
                wider.state_dict()[param_tensor].copy_(toadd3)
            
    return
"""
save_output = SaveOutput()
hook_handles = []

for layer in net.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        handle = layer.register_forward_pre_hook(save_output)
        hook_handles.append(handle)
        print(handle)
"""
new_weights = net_transform_wider(net, wider_net)

x = torch.rand(1, 3, 32, 32)
result = net(x)
result_wider = wider_net(x)
#result_deeper = deeper_net(x)

assert torch.allclose(result,result_wider, atol=1e-05, rtol=0)

