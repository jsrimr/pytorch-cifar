from os import cpu_count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.mobile import MobileNetV2

## Maybe add a little bit of noise as well
## running_mean / running_var
def net_transform_deeper(net, deeper):
    layers_n=[]
    layers_d=[]
    idx = 0

    # Organize in a list
    for name, param in net.named_parameters():
        #print(name, param.size())
        llist = []
        llist.append(name)
        llist.append(param.size())
        layers_n.append(llist)

    for name, param in deeper.named_parameters():
        #print(name, param.size())
        llist = []
        llist.append(name)
        llist.append(param.size())
        layers_d.append(llist)
        print(name)

    for i in range(0, 3):
        weight1 = net.state_dict()[layers_n[idx][0]]
        deeper.state_dict()[layers_d[i][0]].copy_(weight1)
        idx += 1

    for i in range(3, len(layers_d)):
        #print(layers_d[i][0], layers_d[i][1])]
        ly = layers_d[i][0]
        if(not ("layers" in layers_d[i][0])):
            weight1 = net.state_dict()[layers_n[idx][0]]
            deeper.state_dict()[layers_d[i][0]].copy_(weight1)
            idx += 1
            continue
        
        sp = layers_d[i][0].split('.')
        spp = sp[0]+"."+sp[1]+".conv3.weight"
        sp2 = layers_n[idx][0].split('.')
        spp2 = sp2[0]+"."+sp2[1]+".conv3.weight"
        #print("spp : ", spp)
        #print("spp2 : ", spp2)
        #if(layers_d[i][1] == layers_n[idx][1]):
        if (deeper.state_dict()[spp].shape == net.state_dict()[spp2].shape):
            weight1 = net.state_dict()[layers_n[idx][0]]
            deeper.state_dict()[layers_d[i][0]].copy_(weight1)
            idx += 1
        else:
            #print(layers_d[i][0], layers_d[i][1])
            #conv1
            if("conv1" in layers_d[i][0]):
                #a = torch.eye(layers_d[i][1][1]) #Need to check if this works
                a = torch.zeros((layers_d[i][1][1], layers_d[i][1][1]))
                b = torch.unsqueeze(a, 2)
                c = torch.unsqueeze(b, 3)
                ex = round(layers_d[i][1][0] / layers_d[i][1][1])
                cc = c.clone().detach()
                for j in range(1, ex):
                    cc = torch.cat((cc, c), axis=0)
                deeper.state_dict()[layers_d[i][0]].copy_(torch.nn.Parameter(cc)) 
  
            #bn1
            elif("bn" in layers_d[i][0]):
                if("weight" in layers_d[i][0]):
                    c2 = torch.ones(layers_d[i][1])
                else:
                    c2 = torch.zeros(layers_d[i][1])
                    deeper.state_dict()[layers_d[i][0]].copy_(torch.nn.Parameter(c2))              
            #conv2
            elif("conv2" in layers_d[i][0]):
                x = torch.zeros((layers_d[i][1][1], layers_d[i][1][1], 3, 3))
                for j in range(0, layers_d[i][1][1]):
                     x[j][j][1][1] = 1
                deeper.state_dict()[layers_d[i][0]].copy_(torch.nn.Parameter(x))

            #conv3
            elif("conv3" in layers_d[i][0]):
                aa = torch.eye(layers_d[i][1][0])
                bb = torch.unsqueeze(aa, 2)
                cc = torch.unsqueeze(bb, 3)
                ex = round(layers_d[i][1][1] / layers_d[i][1][0])
                ccc = cc.clone().detach()
                for j in range(1, ex):
                    ccc = torch.cat((ccc, cc), axis=1)
                ccc = ccc/ex
                deeper.state_dict()[layers_d[i][0]].copy_(torch.nn.Parameter(ccc))

            #shortcut --> No need to think about this

    ## Last stage copy

    return

newcfg= [(1,  16, 1, 1),
       (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 4, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]

# Real Test
net = MobileNetV2()
deeper_net = MobileNetV2(cfg = newcfg)
new_weights = net_transform_deeper(net, deeper_net)

x = torch.ones((1, 3, 32, 32))
result = net(x)
result_deeper = deeper_net(x)

assert torch.allclose(result,result_deeper, atol=1e-05, rtol=0)