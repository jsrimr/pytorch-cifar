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


def net_transform_wider(net, wider):
    first = 0
    third = 0
    perm = []
    perm3 = []
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
                if(third == 0):
                    diff = wider.state_dict()[param_tensor].shape[0] - net.state_dict()[param_tensor].shape[0]
                    perm3 = torch.randperm(net.state_dict()[param_tensor].shape[0])
                    perm3 = perm3[:diff]
                    third = 1
                    cop = (net.state_dict()[param_tensor]).clone().detach()
                    new = cop.data[perm3]
                    toadd = torch.cat((cop, new), axis=0)
                    wider.state_dict()[param_tensor].copy_(toadd)
                else:
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
                new = 0
                cop = (net.state_dict()[param_tensor]).clone().detach()
                if (("bn3" in param_tensor)):
                    new = cop.data[perm3]
                elif("bn" in param_tensor):
                    new = cop.data[perm]
                else:
                    new = cop.data[perm3]
                toadd = torch.cat((cop, new), axis=0)
                wider.state_dict()[param_tensor].copy_(toadd)
                    
    return 


sml1 = SmallNet()
sml2 = SmallNet2()
#nw = net_transform_wider(sml1, sml2)

x = torch.rand(1,3,32,32)
rs = sml1(x)
rs2 = sml2(x)

#assert torch.allclose(rs,rs2, atol=1e-06, rtol=0)

# Real Test
net = MobileNetV2()
wider_net = wider_MobileNetV2()
#deeper_net = deeper_MobileNetV2()


# function preservation 구현하기
# net_transform_wider , net_transform_deeper function 을 구현해서 아래 test 를 통과해보세요
# 현재 셀의 function call 은 예시일 뿐, 자유롭게 구현하셔도 됩니다. 아래 셀의 test 만 통과하면 됩니다.

new_weights = net_transform_wider(net, wider_net)
#wider_net.load_state_dict(new_weights)

#new_weights = net_transform_deeper(net, deeper_net)
#deeper_net.load_state_dict(new_weights)

#x = torch.rand(1,3,32,32)
x = torch.ones((1, 3, 32, 32))
result = net(x)
result_wider = wider_net(x)
#result_deeper = deeper_net(x)

assert torch.allclose(result,result_wider, atol=1e-06, rtol=0)
#assert torch.allclose(result,result_deeper)



#### REAL TEST

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
#net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
