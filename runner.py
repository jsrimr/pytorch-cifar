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
                noise = torch.Tensor((0.03)*np.random.normal(0,1,new.shape)).cuda()
                #new = new + noise
                cop[:, perm3, :, :] = new
                changed = torch.cat((cop, new), axis=1)

                diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                if (diff > 0):
                    new2 = changed[perm, :, :, :]
                    noise2 = torch.Tensor((0.03)*np.random.normal(0,1,new2.shape)).cuda()
                    #new2 = new2 + noise2
                    changed = torch.cat((changed, new2), axis=0)
                
                ## NOISE
                
                noise = torch.Tensor((0.03)*np.random.normal(0,1,changed.shape)).cuda()
                #changed = changed + noise
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
                        noise3 = torch.Tensor((0.03)*np.random.normal(0,1,new.shape)).cuda()
                        #new = new + noise3
                        toadd = torch.cat((cop, new), axis=0)

                        #NOISE
                        noise = torch.Tensor((0.03)*np.random.normal(0,1,toadd.shape)).cuda()
                        #toadd = toadd + noise
                        wider.state_dict()[param_tensor].copy_(toadd)
                        third = 1
                    else:
                        cop = (net.state_dict()[param_tensor]).clone().detach()
                        new = cop.clone().detach().data[:, perm, :, :]
                        new = new/2
                        noise4 = torch.Tensor((0.03)*np.random.normal(0,1,new.shape)).cuda()
                        #new = new + noise4
                        cop[:, perm, :, :] = new
                        changed = torch.cat((cop, new), axis=1)
                        first = 0
                        df = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                        if (df > 0): #Not last --> Next stage becomes wider as well 
                            perm3_old = perm3.clone().detach()
                            perm3 = torch.randperm(net.state_dict()[param_tensor].shape[0])
                            perm3 = perm3[:df]
                            new2 = changed[perm3, :, :, :]
                            noise5 = torch.Tensor((0.03)*np.random.normal(0,1,new2.shape)).cuda()
                            #new2 = new2 + noise5
                            changed = torch.cat((changed, new2), axis=0)
                        else: #last 
                            third = 0
                        noise = torch.Tensor((0.03)*np.random.normal(0,1,changed.shape)).cuda()
                        #changed = changed + noise
                        wider.state_dict()[param_tensor].copy_(changed)
                    curr_out = c_out
                    
                else: # Same stage as before
                    cop = (net.state_dict()[param_tensor]).clone().detach()
                    new = cop.clone().detach().data[:, perm, :, :]
                    new = new/2
                    noise6 = torch.Tensor((0.03)*np.random.normal(0,1,new.shape)).cuda()
                    #new = new + noise6
                    cop[:, perm, :, :] = new
                    changed = torch.cat((cop, new), axis=1)
                    diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                    if (diff > 0):
                        new2 = changed[perm3, :, :, :]
                        noise7 = torch.Tensor((0.03)*np.random.normal(0,1,new2.shape)).cuda()
                        #new2 = new2 + noise7
                        changed = torch.cat((changed, new2), axis=0)
                    
                    noise = torch.Tensor((0.03)*np.random.normal(0,1,changed.shape)).cuda()
                    #changed = changed + noise
                    wider.state_dict()[param_tensor].copy_(changed)

            elif("conv2" in param_tensor):
                cop3 = (net.state_dict()[param_tensor]).clone().detach()
                new3 = cop3[perm, :, :, :]
                noise8 = torch.Tensor((0.03)*np.random.normal(0,1,new3.shape)).cuda()
                #new3 = new3 + noise8
                toadd3 = torch.cat((cop3, new3), axis=0)
                noise = torch.Tensor((0.03)*np.random.normal(0,1,toadd3.shape)).cuda()
                #toadd3 = toadd3 + noise
                wider.state_dict()[param_tensor].copy_(toadd3)

            else :
                new = torch.Tensor()
                new = new.cuda()
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
                            noise9 = torch.Tensor((0.03)*np.random.normal(0,1,new2.shape)).cuda()
                            #new2 = new2 + noise9
                            cop[:, perm3_old, :, :] = new2
                            cop = torch.cat((cop, new2), axis=1)
                        new = cop.data[perm3]

                    else:
                        new2 = cop.clone().detach().data[:, perm3, :, :]
                        new2 = new2/2
                        noise10 = torch.Tensor((0.03)*np.random.normal(0,1,new2.shape)).cuda()
                        #new2 = new2 + noise10
                        cop[:, perm3, :, :] = new2
                        cop = torch.cat((cop, new2), axis=1)

                else:
                    new = cop.data[perm3]
                #print(param_tensor)
                toadd = torch.cat((cop, new), axis=0)
                wider.state_dict()[param_tensor].copy_(toadd)
                    
    return 

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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    print(best_acc)
    print(start_epoch)


### Change this part as well
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4) #original value : 5e-4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)


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
        torch.save(state, './checkpoint/ckpt20_small_net.pth')
        best_acc = acc

test(0, net)
#new_weights = net_transform_wider(net, wider_net)
#test(0, wider_net)

#ff = open("learningrate.txt", "a")


for epoch in range(start_epoch, start_epoch+200):
    my_lr = scheduler.get_lr()
    run["learning_rate"].log(my_lr)
    print("my_lr : ", my_lr)
    #ff.write(str(my_lr[0])+"\n")
    train(epoch, nt = net)
    test(epoch, net)
    scheduler.step()
 
#ff.close()   


## 1. Change The Loop above (epochs, net name)
## 2. Change optimizer etc (net name, Tmax)
## 3. Change checkpoint name (resume, test())
## 4. network_transform_wider
