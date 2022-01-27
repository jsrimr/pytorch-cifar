import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
from models.mobile import MobileNetV2
from models.mobilenetv2_wider import wider_MobileNetV2
from models.smalltest import SmallNet, SmallNet2
import numpy as np

def net_transform_wider(net, wider):
    first = 0
    third = 0
    perm = []
    perm3 = []
    perm3_old = []
    curr_out = 0
    for param_tensor in net.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        if (net.state_dict()[param_tensor].size() == wider.state_dict()[param_tensor].size()):
            weight1 = net.state_dict()[param_tensor]
            wider.state_dict()[param_tensor].copy_(weight1)
            
        else :
            #print(param_tensor, "\t", wider.state_dict()[param_tensor].size())
            if("conv1" in param_tensor):
                if(first == 0):
                    diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                    perm = torch.randperm(net.state_dict()[param_tensor].shape[0])
                    perm = perm[:diff]
                    first = 1
                cop = (net.state_dict()[param_tensor]).clone().detach()
                new = cop.clone().detach().data[:, perm3, :, :]
                new = new/2
                cop[:, perm3, :, :] = new
                changed = torch.cat((cop, new), axis=1)

                diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                if (diff > 0):
                    new2 = changed[perm, :, :, :]
                    changed = torch.cat((changed, new2), axis=0)

                wider.state_dict()[param_tensor].copy_(changed)
                

            elif("conv3" in param_tensor):
                pt = param_tensor.split('.')
                c_out = net.state_dict()[param_tensor].shape[0]
                if(c_out != curr_out): # stage is different
                    # Beginning
                    if (third ==0):
                        diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                        perm3 = torch.randperm(net.state_dict()[param_tensor].shape[0])
                        perm3 = perm3[:diff]
                        first = 0
                        cop = (net.state_dict()[param_tensor]).clone().detach()
                        new = cop.data[perm3]
                        toadd = torch.cat((cop, new), axis=0)
                        wider.state_dict()[param_tensor].copy_(toadd)
                        third = 1
                    else:
                        cop = (net.state_dict()[param_tensor]).clone().detach()
                        new = cop.clone().detach().data[:, perm, :, :]
                        new = new/2
                        cop[:, perm, :, :] = new
                        changed = torch.cat((cop, new), axis=1)
                        first = 0
                        df = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                        if (df > 0): #Not last --> Next stage becomes wider as well 
                            perm3_old = perm3.clone().detach()
                            perm3 = torch.randperm(net.state_dict()[param_tensor].shape[0])
                            perm3 = perm3[:df]
                            new2 = changed[perm3, :, :, :]
                            changed = torch.cat((changed, new2), axis=0)
                        else: #last 
                            third = 0
                        wider.state_dict()[param_tensor].copy_(changed)
                    curr_out = c_out
                    
                else: # Same stage as before
                    cop = (net.state_dict()[param_tensor]).clone().detach()
                    new = cop.clone().detach().data[:, perm, :, :]
                    new = new/2
                    cop[:, perm, :, :] = new
                    changed = torch.cat((cop, new), axis=1)
                    diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                    if (diff > 0):
                        new2 = changed[perm3, :, :, :]
                        changed = torch.cat((changed, new2), axis=0)

                    wider.state_dict()[param_tensor].copy_(changed)

            elif("conv2" in param_tensor):
                cop3 = (net.state_dict()[param_tensor]).clone().detach()
                new3 = cop3[perm, :, :, :]
                toadd3 = torch.cat((cop3, new3), axis=0)
                wider.state_dict()[param_tensor].copy_(toadd3)

            else :
                new = torch.Tensor()
                cop = (net.state_dict()[param_tensor]).clone().detach()
                if (("bn3" in param_tensor)):
                    new = cop.data[perm3]
                elif("bn" in param_tensor):
                    new = cop.data[perm]
                elif("0.weight" in param_tensor):
                    aa = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                    bb = wider.state_dict()[param_tensor].shape[1] - net.state_dict()[param_tensor].shape[1]
                    if(aa > 0):
                        if (bb > 0):
                            new2 = cop.clone().detach().data[:, perm3_old, :, :]
                            new2 = new2/2
                            cop[:, perm3_old, :, :] = new2
                            cop = torch.cat((cop, new2), axis=1)
                        new = cop.data[perm3]

                    else:
                        new2 = cop.clone().detach().data[:, perm3, :, :]
                        new2 = new2/2
                        cop[:, perm3, :, :] = new2
                        cop = torch.cat((cop, new2), axis=1)

                else:
                    new = cop.data[perm3]
                toadd = torch.cat((cop, new), axis=0)
                #print(param_tensor)
                wider.state_dict()[param_tensor].copy_(toadd)
                    
    return

def shuffle(net):
    for param_tensor in net.state_dict():
        print(param_tensor)
        a = (net.state_dict()[param_tensor]).clone().detach().numpy()
        if(a.ndim <2):
            continue
        gh = a.reshape((a.size,))
        np.random.shuffle(gh)
        gh = torch.Tensor(gh.reshape(a.shape))
        print(gh.shape)
        net.state_dict()[param_tensor].copy_(gh)
    return

def part_shuffle(net, pct):
    for param_tensor in net.state_dict():
        a = (net.state_dict()[param_tensor]).clone().detach()
        b = a.clone().detach().cpu().numpy()
        if(b.ndim <2):
            continue
        nn = round(net.state_dict()[param_tensor].shape[0] * pct)
        perm = torch.randperm(net.state_dict()[param_tensor].shape[0])
        i = 0
        while(i+1 < nn):
            aa = a[perm[i]]
            a[perm[i]] = a[perm[i+1]]
            a[perm[i+1]] = aa
            i +=2
        net.state_dict()[param_tensor].copy_(a)

sml1 = SmallNet()
sml2 = SmallNet2()
#nw = net_transform_wider(sml1, sml2)

x = torch.rand(1,3,32,32)
rs = sml1(x)
rs2 = sml2(x)

#assert torch.allclose(rs,rs2, atol=1e-06, rtol=0)
newcfg= [(1,  20, 1, 1),
       (6,  26, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  42, 3, 2),
       (6,  70, 4, 2),
       (6,  132, 3, 1),
       (6, 180, 3, 2),
       (6, 320, 1, 1)]

# Real Test
net = MobileNetV2()
wider_net = MobileNetV2(cfg = newcfg)

new_weights = net_transform_wider(net, wider_net)
part_shuffle(wider_net, 0.2)

x = torch.ones((1, 3, 32, 32))
result = net(x)
result_wider = wider_net(x)
#result_deeper = deeper_net(x)

assert torch.allclose(result,result_wider, atol=1e-05, rtol=0)
#assert torch.allclose(result,result_deeper)