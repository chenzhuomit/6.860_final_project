import torch
from torch import nn
from torch.nn import functional as F

from .resnet import ResNet

class SharedResBlock(nn.Module):
    def __init__(self, hidden_channels, kernel_size, batch_norm=True, resnet=True, dropout=0.):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else (kernel_size // 2, kernel_size // 2 - 1)
        self.batch_norm = batch_norm
        self.resnet = resnet
        self.dropout = dropout
        layers = []
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        self.resblock = ResNet(nn.Sequential(*layers), resnet=resnet)
    
    def forward(self, x):
        return F.leaky_relu(self.resblock(x))
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

class SharedConvLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size1, kernel_size2, kernel_size3, batch_norm=True, pooling=True, resnet=True, dropout=0.):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.padding1 = kernel_size1 // 2 if kernel_size1 % 2 == 1 else (kernel_size1 // 2, kernel_size1 // 2 - 1)
        self.padding3 = kernel_size3 // 2 if kernel_size1 % 2 == 1 else (kernel_size3 // 2, kernel_size3 // 2 - 1)
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.resnet = resnet
        self.dropout = dropout
        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_channels, self.kernel_size1, padding=self.padding1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        if pooling:
            layers.append(nn.MaxPool2d(2))
        layers.append(SharedResBlock(hidden_channels, self.kernel_size2, batch_norm, resnet, dropout))
        if pooling:
            layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(hidden_channels, out_channels, self.kernel_size3, padding=self.padding3))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        if pooling:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    
class SharedConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size1, kernel_size2, kernel_size3, intermediate_dim, nb_classes, batch_norm=True, pooling=True, resnet=True, dropout=0., convdropout=0.):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.intermidiate_dim = intermediate_dim
        self.nb_classes = nb_classes
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.resnet = resnet
        self.dropout = dropout
        self.convdropout = convdropout
        self.convlayer = SharedConvLayer(in_channels, hidden_channels, out_channels, kernel_size1, kernel_size2, kernel_size3, batch_norm, pooling, resnet, convdropout)
        if pooling:
            self.conv_out_dim = 4 * 4 * out_channels
        else:
            self.conv_out_dim = 32 * 32 * out_channels
        fflayer = []
        fflayer.append(nn.Linear(self.conv_out_dim, intermediate_dim))
        fflayer.append(nn.Dropout(dropout))
        fflayer.append(nn.LeakyReLU())
        fflayer.append(nn.Linear(intermediate_dim, nb_classes))
        self.fflayer = nn.Sequential(*fflayer)
        
        
    def forward(self, x):
        y = self.convlayer(x)
        y = y.view(y.shape[0], -1)
        return self.fflayer(y)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
        
        