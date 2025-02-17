'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.expansion = expansion
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:  # stage 의 첫블럭인 경우 이게 필요함
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


CFG = [(1, 16, 1, 1),
       (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6, 32, 3, 2),
       (6, 64, 4, 2),
       (6, 96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    def __init__(self, num_classes=10, cfg=CFG):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(cfg[-1][1], 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        # layers = []
        module_list = [] #nn.Sequential()
        for stage, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            stage_blocks = nn.Sequential()
            strides = [stride] + [1] * (num_blocks - 1)
            for block_idx, stride in enumerate(strides):
                # layers.append(Block(in_planes, out_planes, expansion, stride))
                stage_blocks.add_module(f"stage_{stage}_block{block_idx}", Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
            module_list.append(stage_blocks)

        return nn.Sequential(*module_list, )
        # return module_list

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layers(out)
        for l in self.layers:
            out = l(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
