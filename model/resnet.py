import torch
from torch import nn
from torch.nn import functional as F


class ResNet(nn.Module):
    def __init__(self, net, resnet=True):
        super().__init__()
        self.net = net
        self.resnet = resnet
    def forward(self, x):
        if self.resnet:
            identity = x
        else:
            identity = 0.
        return self.net(x) + identity