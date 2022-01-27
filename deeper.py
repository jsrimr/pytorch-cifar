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
       (6,  24, 3, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 4, 2),
       (6,  64, 6, 2),
       (6,  96, 4, 1),
       (6, 160, 4, 2),
       (6, 320, 1, 1)]

# Real Test

#new_weights = net_transform_deeper(net, deeper_net)

#x = torch.ones((1, 3, 32, 32))
#result = net(x)
#result_deeper = deeper_net(x)

#assert torch.allclose(result,result_deeper, atol=1e-05, rtol=0)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


run = neptune.init(
        project="caplab/net-transform-train-setting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODE0YWMzMC02MWE2LTQ4MmUtODQ0MS0yZWM1NGI3MDYwNzQifQ==",
    )

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

net = MobileNetV2()
deeper_net = MobileNetV2(cfg = newcfg)
#new_weights = net_transform_deeper(net, deeper_net)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


deeper_net = deeper_net.to(device)
if device == 'cuda':
    deeper_net = torch.nn.DataParallel(deeper_net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt08_150epoch.pth')
    net.load_state_dict(checkpoint['net'])
    #for param_tensor in net.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        #new_t = param_tensor
        #net.state_dict()[param_tensor] = checkpoint['net'][new_t]
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(best_acc)
    print(start_epoch)


### Change this part as well
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(deeper_net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4) #original value : 5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)


# Training
def train(epoch, nt = net):
    print('\nEpoch: %d' % epoch)
    nt.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = nt(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, nt):
    global best_acc
    nt.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = nt(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            run["eval/loss"].log(loss.item())
            run["eval/acc"].log(100. * correct / total)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': nt.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt11_deep_300epoch.pth')
        best_acc = acc

#test(0, net)
#new_weights = net_transform_wider(net, wider_net)
#test(0, wider_net)

ff = open("learningrate_deep.txt", "a")



for epoch in range(start_epoch, start_epoch+300):
    my_lr = scheduler.get_lr()
    run["learning_rate"].log(my_lr)
    print("my_lr : ", my_lr)
    ff.write(str(my_lr[0])+"\n")
    train(epoch, nt = deeper_net)
    test(epoch, deeper_net)
    scheduler.step()
 
ff.close()   

## 1. Change The Loop above (epochs, net name)
## 2. Change optimizer etc (net name, Tmax)
## 3. Change checkpoint name (resume, test())
## 4. network_transform_wider