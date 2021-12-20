'''Train CIFAR10 with PyTorch.'''

import torchvision
import torchvision.transforms as transforms

import argparse

from models import *
from utils import block_update, process_bn_weight, get_wider_weight

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

from models.mobilenetv2 import Block

from utils import BASE_CFG, net_transform_wider_update, BlockWeight


def check_weight_sync(weight_name, parent_dict, child_dict, axis):
    parent_w = parent_dict[weight_name]
    child_w = child_dict[weight_name]

    subset = parent_w.shape[axis]
    if axis == 0:
        assert torch.allclose(parent_w, child_w[:subset, :, :, :])
    elif axis == 1:
        assert torch.allclose(parent_w, child_w[:, :subset, :, :])


def test_net_level(parent_cfg, child_cfg, parent_ckpt):
    # Model
    print('==> Building model..')

    parent_net = MobileNetV2(cfg=parent_cfg)
    checkpoint = torch.load(parent_ckpt)
    parent_net.load_state_dict(checkpoint['net'])
    # net = net.to(device)

    child_net = MobileNetV2(cfg=child_cfg)
    net_transform_wider_update(parent_net, child_net, 4)

    # todo : 맨 마지막 스테이지 처리

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
        print(f"child accuracy = {100. * correct / total}")

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


def test_block_level():
    block1 = Block(3, 16, 6, 1)
    block2 = Block(16, 32, 6, 1)

    wider_block1 = Block(3, 20, 6, 1)
    wider_block2 = Block(20, 32, 6, 1)

    x = torch.rand(1, 3, 28, 28)

    res1 = block1(x)
    res1 = block2(res1)

    res2 = wider_block1(x)
    res2 = wider_block2(res2)

    print(f"Before : res1==res2 : {torch.allclose(res1, res2[:,:res1.shape[1], :, :], rtol=1e-03, atol=1e-04)}")

    new_dict1, new_dict2 = block_update(block1, block2, wider_block1, wider_block2)
    wider_block1.load_state_dict(new_dict1)
    wider_block2.load_state_dict(new_dict2)


    res1 = block1(x)
    res2 = wider_block1(x)
    # assert torch.allclose(res1, res2[:, :16, :, :])

    res1 = block2(res1)
    res2 = wider_block2(res2)

    # assert torch.allclose(res1, res2, rtol=1e-03, atol=1e-04)
    print(f"After : res1==res2 : {torch.allclose(res1, res2[:,:res1.shape[1], :, :], rtol=1e-03, atol=1e-04)}")


def test_layer_level():
    x = torch.rand(1, 3, 28, 28)

    p_conv = nn.Conv2d(3, 5, 1, bias=False)
    p_bn1 = nn.BatchNorm2d(5)
    p_conv_2 = nn.Conv2d(5, 1, 1, bias=False)
    p_bn2 = nn.BatchNorm2d(1)
    res1 = p_bn2(p_conv_2(p_bn1(p_conv(x))))

    c_conv = nn.Conv2d(3, 10, 1, bias=False)
    c_bn1 = nn.BatchNorm2d(10)
    c_conv_2 = nn.Conv2d(10, 1, 1, bias=False)
    c_bn2 = nn.BatchNorm2d(1)
    res2 = c_bn2(c_conv_2(c_bn1(c_conv(x))))

    print(f"res1==res2 : {torch.allclose(res1, res2)}")

    conv_block1_wider = get_wider_weight(p_conv.weight.detach().numpy(), c_conv.weight.detach().numpy(), axis=0)
    wider_bn1 = process_bn_weight(p_bn1, c_bn1)
    conv_block2_wider = get_wider_weight(p_conv_2.weight.detach().numpy(), c_conv_2.weight.detach().numpy(),
                                         axis=1)
    wider_bn2 = process_bn_weight(p_bn2, c_bn2)

    c_conv.weight.data = torch.from_numpy(conv_block1_wider)
    c_conv_2.weight.data = torch.from_numpy(conv_block2_wider)

    c_bn1.weight.data = torch.from_numpy(wider_bn1[:, 0])
    c_bn1.bias.data = torch.from_numpy(wider_bn1[:, 1])
    c_bn1.running_mean.data = torch.from_numpy(wider_bn1[:, 2])
    c_bn1.running_var.data = torch.from_numpy(wider_bn1[:, 3])
    c_bn1.num_batches_tracked.data = p_bn1.num_batches_tracked

    c_bn2.weight.data = torch.from_numpy(wider_bn2[:, 0])
    c_bn2.bias.data = torch.from_numpy(wider_bn2[:, 1])
    c_bn2.running_mean.data = torch.from_numpy(wider_bn2[:, 2])
    c_bn2.running_var.data = torch.from_numpy(wider_bn2[:, 3])
    c_bn2.num_batches_tracked.data = p_bn2.num_batches_tracked

    res1 = p_bn2(p_conv_2(p_bn1(p_conv(x))))
    res2 = c_bn2(c_conv_2(c_bn1(c_conv(x))))

    assert torch.allclose(res1, res2)


if __name__ == '__main__':
    # test_layer_level()
    # test_block_level()

    parent_ckpt = './checkpoint/base_ckpt.pth'

    parent_cfg = BASE_CFG

    child_cfg = [[1, 16, 1, 1],
                 [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                 [6, 16, 2, 2],
                 [6, 32, 3, 2],
                 [6, 64, 2, 1],  # 48->64
                 [6, 96, 2, 2],
                 [6, 320, 1, 1]]

    test_net_level(parent_cfg, child_cfg, parent_ckpt)

    # load parent

    # resume train
