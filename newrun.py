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
        

def net_transform_wider(net, wider):
    for param_tensor in net.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        if (net.state_dict()[param_tensor].size() == wider.state_dict()[param_tensor].size()):
            weight1 = net.state_dict()[param_tensor]
            wider.state_dict()[param_tensor].copy_(weight1)
            
        else :
            #print(param_tensor, "\t", wider.state_dict()[param_tensor].size())
            if(("conv" in param_tensor) or ("shortcut.0" in param_tensor)):
                diff1 = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                diff2 = wider.state_dict()[param_tensor].shape[1] - net.state_dict()[param_tensor].shape[1]
                toadd= torch.Tensor().cuda()
                cop = (net.state_dict()[param_tensor]).clone().detach()
                mn = torch.mean(cop)
                vr = torch.std(cop)
                if((diff1 > 0)):
                    # In between
                    sh = list(net.state_dict()[param_tensor].shape)
                    sh[0] = diff1
                    new = torch.normal(mn,vr,size = sh).to(device)
                    cop = torch.cat((cop, new), axis=0)
                    toadd = cop.clone().detach()
                    
                #end
                if(diff2>0):
                    sh2 = list(wider.state_dict()[param_tensor].shape)
                    sh2[1] = diff2
                    new2 = torch.zeros(sh2).cuda()
                    new2 = new2 + torch.normal(mn,vr,size = sh2).to(device)
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
ncfg= [(1,  16, 1, 1),
       (6,  30, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  40, 3, 2),
       (6,  80, 4, 2),
       (6, 120, 3, 1),
       (6, 200, 3, 2),
       (6, 320, 1, 1)]

# Real Test
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(best_acc)
    print(start_epoch)


### Change this part as well
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(wider_net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4) #original value : 5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10)


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
        torch.save(state, './checkpoint/ckpt13_wide_middlestart.pth')
        best_acc = acc

#test(0, net)
new_weights = net_transform_wider(net, wider_net)
test(0, wider_net)


"""
for hh in range(0, 100):
    optimizer.step()
    scheduler.step()
"""

for epoch in range(start_epoch, start_epoch+100):
    my_lr = scheduler.get_lr()
    run["learning_rate"].log(my_lr)
    print("my_lr : ", my_lr)
    train(epoch, nt = wider_net)
    test(epoch, wider_net)
    scheduler.step()
 

