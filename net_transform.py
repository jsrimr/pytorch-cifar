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
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
total_epoch = 201

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

from utils import get_next_net, train, test, BASE_CFG

if __name__ == '__main__':

    PLAN = [
        ("width", 2),
        ("depth", 2),
        ("width", 3),
        ("depth", 3),
        ("width", 4),
        ("depth", 4),
        ("width", 5),
        ("depth", 5),
    ]

    run = neptune.init(
        project="caplab/net-transform-train-setting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MDQyMjIyZC1mYWQ0LTQ3OWYtYmY1Ny0yMmZlNzA0ODg5NzkifQ==",
    )  # your credentials

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    # if FROM_SCRATCH:  # train from scratch => todo : cifar 에서 별 차이 없으면 Imagenet-100 으로 가자
    # else:
    #
    #     # net = MobileNetV2(cfg=CFG).to(device)

    net = MobileNetV2(cfg=BASE_CFG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/b_a_s_e_ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print("loaded the model")

    plan_idx = None
    interval = total_epoch // len(PLAN)
    for epoch in range(start_epoch, total_epoch):
        if epoch % interval == 0:
            plan_idx = epoch // interval - 1
            net = get_next_net(net, PLAN, plan_idx).to(device)
            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            print("=" * 100 + "\n", f"net changed! {PLAN[plan_idx]}" + "\n", "=" * 100)

        train(net, criterion, optimizer, trainloader, epoch, device, run)

        exp_name = '_'.join(map(str, PLAN[plan_idx])) if plan_idx else "base"
        test(net, criterion, testloader, epoch, device, run, exp_name)
        scheduler.step()

    run.stop()
