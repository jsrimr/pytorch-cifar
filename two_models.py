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

ncfg= [(1,  16, 1, 1),
       (6,  30, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  40, 3, 2),
       (6,  80, 4, 2),
       (6, 120, 3, 1),
       (6, 200, 3, 2),
       (6, 320, 1, 1)]


net = MobileNetV2()
wider_net = MobileNetV2(cfg = ncfg)
net2 = MobileNetV2()
wider_net2 = MobileNetV2(cfg = ncfg)
net3 = MobileNetV2()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net2 = net2.to(device)
if device == 'cuda':
    net2 = torch.nn.DataParallel(net2)
    cudnn.benchmark = True

net3 = net3.to(device)
if device == 'cuda':
    net3 = torch.nn.DataParallel(net3)
    cudnn.benchmark = True

wider_net = wider_net.to(device)
if device == 'cuda':
    wider_net = torch.nn.DataParallel(wider_net)
    cudnn.benchmark = True


wider_net2 = wider_net2.to(device)
if device == 'cuda':
    wider_net2 = torch.nn.DataParallel(wider_net2)
    cudnn.benchmark = True

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print(best_acc)
print(start_epoch)

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint2 = torch.load('./checkpoint/ckpt13_small_net.pth')
net2.load_state_dict(checkpoint2['net'])

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint3 = torch.load('./checkpoint/ckpt20_small_net.pth')
net3.load_state_dict(checkpoint3['net'])


def twonets(n1, n2):
    for param_tensor in n1.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        nn1 = (n1.state_dict()[param_tensor]).clone().detach()
        nn2 = (n2.state_dict()[param_tensor]).clone().detach()
        nn1 = (nn1 + nn2)/2.0
        n1.state_dict()[param_tensor].copy_(nn1)

def threenets_rand_lyr(n1, n2, n3):
    lyr = ''
    a = 0
    for param_tensor in n1.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        nn1 = 0
        if("layer" in param_tensor):
            idx = param_tensor.index('layer')
            num = param_tensor[idx+5:idx+7]
            if(num != lyr):
                lyr = num
                a = torch.rand(1)
            nn1 = ((n1.state_dict()[param_tensor]).clone().detach() + (n2.state_dict()[param_tensor]).clone().detach())/2.0
            if (a > 2.0/3.0):
                nn1 = ((n2.state_dict()[param_tensor]).clone().detach() + (n3.state_dict()[param_tensor]).clone().detach())/2.0
            elif (a > 1.0/3.0):
                nn1 = ((n3.state_dict()[param_tensor]).clone().detach() + (n1.state_dict()[param_tensor]).clone().detach())/2.0
        
        else:
            nn1 = (n1.state_dict()[param_tensor]).clone().detach()
            nn2 = (n2.state_dict()[param_tensor]).clone().detach()
            nn3 = (n3.state_dict()[param_tensor]).clone().detach()
            nn1 = (nn1 + nn2 +nn3)/3.0
        n1.state_dict()[param_tensor].copy_(nn1)

def threenets_lyr_2(n1, n2, n3):
    lyr = ''
    a = 0
    for param_tensor in n1.state_dict():
        #print(param_tensor, "\t", net.state_dict()[param_tensor].size())
        nn1 = 0
        if("layer" in param_tensor):
            idx = param_tensor.index('layer')
            num = param_tensor[idx+5:idx+7]
            if(num != lyr):
                lyr = num
                a = torch.rand(1)
            nn1 = ((n1.state_dict()[param_tensor]).clone().detach() + (n2.state_dict()[param_tensor]).clone().detach())/2.0
            if (a > 1.0/2.0):
                nn1 = ((n2.state_dict()[param_tensor]).clone().detach() + (n3.state_dict()[param_tensor]).clone().detach())/2.0
        
        else:
            nn1 = (n1.state_dict()[param_tensor]).clone().detach()
            nn2 = (n2.state_dict()[param_tensor]).clone().detach()
            nn3 = (n3.state_dict()[param_tensor]).clone().detach()
            nn1 = (nn1 + nn2 +nn3)/3.0
        n1.state_dict()[param_tensor].copy_(nn1)


### Change this part as well
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(wider_net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-3) #original value : 5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10)

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
                new = torch.Tensor().cuda()
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        run["eval/loss"].log(test_loss)
        run["eval/acc"].log(100. * correct / total)
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
        torch.save(state, './checkpoint/ckpt20_twomodel150.pth')
        best_acc = acc

#test(0, net)
#twonets(net, net2)
threenets_lyr_2(net, net2, net3)
new_weights = net_transform_wider(net, wider_net)
test(0, wider_net)


"""
for hh in range(0, 100):
    optimizer.step()
    scheduler.step()
"""

for epoch in range(start_epoch, start_epoch+200):
    my_lr = scheduler.get_lr()
    run["learning_rate"].log(my_lr)
    print("my_lr : ", my_lr)
    #ff.write(str(my_lr[0])+"\n")
    train(epoch, nt = wider_net)
    test(epoch, wider_net)
    scheduler.step()

"""
scheduler2.step()
for epoch in range(start_epoch +100, start_epoch+150):
    my_lr = scheduler2.get_lr()
    run["learning_rate"].log(my_lr)
    print("my_lr : ", my_lr)
    #ff.write(str(my_lr[0])+"\n")
    train(epoch, nt = wider_net)
    test(epoch, wider_net)
    scheduler2.step()
"""

## 1. Change The Loop above (epochs, net name)
## 2. Change optimizer etc (net name, Tmax)
## 3. Change checkpoint name (resume, test())
## 4. network_transform_wider