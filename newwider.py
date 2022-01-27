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

def net_transform_wider(net, wider):
    for param_tensor in net.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        if (net.state_dict()[param_tensor].size() == wider.state_dict()[param_tensor].size()):
            weight1 = net.state_dict()[param_tensor]
            wider.state_dict()[param_tensor].copy_(weight1)
            
        else :
            #sprint(param_tensor, "\t", wider.state_dict()[param_tensor].size())
            if(("conv" in param_tensor) or ("shortcut.0" in param_tensor)):
                diff1 = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                diff2 = wider.state_dict()[param_tensor].shape[1] - net.state_dict()[param_tensor].shape[1]
                toadd= torch.Tensor()
                cop = (net.state_dict()[param_tensor]).clone().detach()
                if((diff1 > 0)):
                    # In between
                    sh = list(net.state_dict()[param_tensor].shape)
                    sh[0] = diff1
                    new = (-1) + 2*torch.rand(sh)
                    cop = torch.cat((cop, new), axis=0)
                    toadd = cop.clone().detach()
                    
                #end
                if(diff2>0):
                    sh2 = list(wider.state_dict()[param_tensor].shape)
                    sh2[1] = diff2
                    new2 = torch.zeros(sh2)
                    toadd = torch.cat((cop, new2), axis=1)
                wider.state_dict()[param_tensor].copy_(toadd)

            else:
                cop3 = (net.state_dict()[param_tensor]).clone().detach()
                diff3 = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                new3 = cop3.data[:diff3]
                toadd3 = torch.cat((cop3, new3), axis=0)
                wider.state_dict()[param_tensor].copy_(toadd3)
            
    return 



#assert torch.allclose(rs,rs2, atol=1e-06, rtol=0)
newcfg= [(1,  16, 1, 1),
       (6,  30, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  40, 3, 2),
       (6,  80, 4, 2),
       (6, 120, 3, 1),
       (6, 200, 3, 2),
       (6, 320, 1, 1)]

# Real Test
net = MobileNetV2()
#wider_net = wider_MobileNetV2()
wider_net = MobileNetV2(cfg = newcfg)
#deeper_net = deeper_MobileNetV2()


# function preservation 구현하기
# net_transform_wider , net_transform_deeper function 을 구현해서 아래 test 를 통과해보세요
# 현재 셀의 function call 은 예시일 뿐, 자유롭게 구현하셔도 됩니다. 아래 셀의 test 만 통과하면 됩니다.

new_weights = net_transform_wider(net, wider_net)

x = torch.rand(1,3,32,32)
#x = torch.ones((1, 3, 32, 32))
result = net(x)
result_wider = wider_net(x)
#result_deeper = deeper_net(x)

assert torch.allclose(result,result_wider, atol=1e-06, rtol=0)
#assert torch.allclose(result,result_deeper)
