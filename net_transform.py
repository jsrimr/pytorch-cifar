import os
import argparse

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar

import neptune.new as neptune

FROM_SCRATCH = True
from models.mobilenetv2 import MobileNetV2


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

from utils import train, test, BASE_CFG

if __name__ == '__main__':
    run = neptune.init(
        project="caplab/net-transform-train-setting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MDQyMjIyZC1mYWQ0LTQ3OWYtYmY1Ny0yMmZlNzA0ODg5NzkifQ==",
    )  # your credentials

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    if FROM_SCRATCH:  # train from scratch => todo : cifar 에서 별 차이 없으면 Imagenet-100 으로 가자
        net = MobileNetV2(cfg=BASE_CFG).to(device)

    else:
        # target cfg
        pass
        # CFG = [
        #     (1, 16, 1, 1),
        #     (6, 24, 2, 1),
        #     (6, 32, 3, 2), #16->32 , 2->3
        #     (6, 48, 5, 2), #32->64 , 4->5
        #     (6, 96, 4, 1), #48->96 , 3->4
        #     (6, 192, 3, 2), #96->192, 2->3
        #     (6, 380, 1, 1)]
        #
        #
        # net = MobileNetV2(cfg=CFG).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(1, 201):
        # if epoch % 25 == 0:
        #     net = get_next_net(net, plan, epoch)


        train(net, criterion, optimizer, trainloader, epoch, device, run)
        test(net, criterion, testloader, epoch, device, run, "base")
        scheduler.step()

    run.stop()
