'''Train CIFAR10 with PyTorch.'''
import argparse

import neptune.new as neptune

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim

from net_transform import make_scheduler, get_warmed_new_scheduler, PLAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

from models.mobilenetv2 import Block, MobileNetV2

from utils import BASE_CFG, process_bn, get_next_net, train, test


def test_net_level():
    parent_net = MobileNetV2(cfg=BASE_CFG)
    checkpoint = torch.load('./checkpoint/base_ckpt.pth')
    parent_net.load_state_dict(checkpoint['net'])

    parent_net.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = parent_net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f"parent accuracy = {100. * correct / total}")

    deeper_net_cfg = [[1, 16, 1, 1],
                      [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                      [6, 16, 2, 2],
                      [6, 32, 4, 2],
                      [6, 48, 3, 1],
                      [6, 96, 3, 2],  # 2=>3
                      [6, 320, 1, 1]]

    child_net = MobileNetV2(cfg=deeper_net_cfg)
    child_net.load_state_dict(parent_net.state_dict(), strict=False)
    child_net.layers[5][-1].bn3 = process_bn(child_net.layers[5][-1].bn3)

    child_net.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = child_net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f"deeper accuracy = {100. * correct / total}")


def test_block_level():
    block1 = Block(3, 16, 6, 1)

    exp_block1 = Block(3, 16, 6, 1)
    exp_block2 = Block(16, 16, 6, 1)
    exp_block1.load_state_dict(block1.state_dict())

    x = torch.rand(1, 3, 28, 28)

    res1 = block1(x)
    res2 = exp_block1(x)
    res2 = exp_block2(res2)

    print(f"Before res1==res2 : {torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)}")

    # exp_block2 update
    # conv1_w_identity = get_conv_identity_layer(exp_block2.conv1)
    #
    # conv2_w_identity = get_conv_identity_layer(exp_block2.conv2)
    #
    # conv3_w_identity = get_conv_identity_layer(exp_block2.conv3)

    # exp_block2.conv1.weight.data = conv1_w_identity
    # exp_block2.conv2.weight.data = conv2_w_identity
    # exp_block2.bn2 = bn2_identity
    #
    # exp_block2.conv3.weight.data = conv3_w_identity
    # exp_block2.bn3 = bn3_identity
    # exp_block2.conv1(test_input)[:, :test_input.shape[1], :, :] == test_input

    # test_input = torch.rand(1, 96, 32, 32)
    #
    # # test_input = torch.rand(1, 3, 2, 2)
    # test_bn = nn.BatchNorm2d(3)
    # test_bn(test_input)
    #
    # test_bn.running_mean * test_bn.momentum + test_input.mean(dim=[0, 2, 3]) * (1 - test_bn.momentum)
    # test_bn(test_input)
    # test_bn.running_mean
    #
    # exp_block2.bn1(test_input) == test_input
    #
    # exp_block2.train()
    # exp_block2.bn1 = bn_identity(exp_block2.bn1)
    # exp_block2.bn1.running_mean == exp_block2.bn1.bias
    # exp_block2.bn1.weight == torch.sqrt(exp_block2.bn1.running_var + exp_block2.bn1.eps)
    # exp_block2.eval()
    # exp_block2.bn1(test_input) == test_input
    #
    # exp_block2.bn1(exp_block2.conv1(test_input))[:, :test_input.shape[1], :, :] == test_input

    bn3_identity = process_bn(exp_block2.bn3)
    exp_block2.bn3 = bn3_identity

    res1 = block1(x)

    res2 = exp_block1(x)
    res2 = exp_block2(res2)

    print(f"After res1==res2 : {torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)}")
    assert torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)


def test_layer_level():
    pass


if __name__ == '__main__':
    # test_layer_level()
    # test_block_level()
    # test_net_level()

    run = neptune.init(
        project="caplab/net-transform-train-setting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MDQyMjIyZC1mYWQ0LTQ3OWYtYmY1Ny0yMmZlNzA0ODg5NzkifQ==",
    )  # your credentials

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    current_cfg = BASE_CFG
    net = MobileNetV2(cfg=current_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    checkpoint = torch.load('./checkpoint/tmp_ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    scheduler = make_scheduler(optimizer)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # NOTE : 처음에는 직접 load_state_dict 해줘야함. get_warmed_scheduler(scheduler) 에서 scheduler 는 준비된 scheduler 를 요구하기 때문
    new_sched_state_dict = {"last_epoch": checkpoint['scheduler']['last_epoch'],
                            "_step_count": checkpoint['scheduler']['_step_count'],
                            "_last_lr": checkpoint['scheduler']['_last_lr']}
    scheduler.load_state_dict(new_sched_state_dict)

    print("loaded the model")

    plan_idx = 1
    target, stage = PLAN[plan_idx]
    net, current_cfg = get_next_net(net, current_cfg, PLAN, plan_idx)
    print("=" * 20 + "\n", f"net changed! {PLAN[plan_idx]}\n", "=" * 20)
    net = net.to(device)  # NOTE : net should be on the right device before loading optimizer, scheduler

    optimizer = optim.SGD(net.parameters(), lr=optimizer.param_groups[0]['lr'],
                          momentum=0.9, weight_decay=5e-4)

    scheduler = get_warmed_new_scheduler(scheduler, optimizer)

    plan_idx = None

    total_epoch = 200
    start_epoch = 0
    for epoch in range(start_epoch, total_epoch):
        train(net, criterion, optimizer, trainloader, epoch, device, run)

        exp_name = "depth_2_full"
        test(net, optimizer, scheduler, criterion, testloader, epoch, device, run, exp_name)

        run['train/lr'].log(scheduler.get_last_lr())
        scheduler.step()

    run.stop()
