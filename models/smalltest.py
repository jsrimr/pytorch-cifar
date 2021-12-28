import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):

    def __init__(self, num_classes=10):
        super(SmallNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.conv3 = nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.shortcut = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(4),
            )
        self.conv4 = nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        
            

    def forward(self, x):
        out = self.conv1(x)
        out2 = F.relu(self.bn1(out))
        out2 = self.conv2(out2)
        out2 = self.conv3(out2)
        out2 = F.relu(self.bn2(out2))
        out2 = out2 + self.shortcut(x)
        out2 = self.conv4(out2)
        return out2

class SmallNet2(nn.Module):

    def __init__(self, num_classes=10):
        super(SmallNet2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.conv3 = nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(6)
        self.shortcut = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(6),
            )
        self.conv4 = nn.Conv2d(6, 8, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        

    def forward(self, x):
        x2 = self.conv1(x)
        out = F.relu(self.bn1(x2))
        out = self.conv2(out)
        out = self.conv3(out)
        out = F.relu(self.bn2(out))
        out = out + self.shortcut(x)
        #out = F.relu(self.bn2(self.conv3(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = self.conv4(out)
        return out