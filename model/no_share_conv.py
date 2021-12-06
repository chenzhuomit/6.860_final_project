import math
import torch
from torch import nn
from torch.nn import functional as F

from .resnet import ResNet

class NoShareConv2d(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        if type(stride) == int:
            stride = (stride, stride)
        if type(padding) == int:
            padding = (padding, padding)
        if type(dilation) == int:
            dilation = (dilation, dilation)
        assert stride == (1, 1), 'only implemented stride is 1 case'
        assert dilation == (1, 1), 'only implemented dilation is 1 case'
        assert groups == 1, 'only implemented group is 1 case'
        assert padding_mode == 'zeros', 'only implemented zero padding'
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.in_dim = in_channels * kernel_size**2
        self.length = (input_size + 2 * padding[0] - (kernel_size-1))
        self.out_dim = out_channels
        
        self.unfoldlayer = nn.Unfold(kernel_size, dilation, padding, stride)
        self.weight = nn.Parameter(torch.Tensor(self.length**2, self.in_dim, self.out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.length**2, self.out_dim))
        else:
            self.bias = None
        self.reset_parameters()
        # self.fflayers = nn.ModuleList([
        #     nn.Linear(self.in_dim, self.out_dim, bias) for i in range(self.length**2)
        #     ])
        
    def forward(self, x):
        y = self.unfoldlayer(x).permute(2, 0, 1).contiguous()
        y = torch.bmm(y, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(1)
        y = y.permute(1, 2, 0).contiguous()
        return y.view(y.shape[0], self.out_channels, self.length, self.length)
    
    def reset_parameters(self):
        for i in range(self.length**2):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)
                
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
    
    
class NoShareResBlock(nn.Module):
    def __init__(self, input_size, hidden_channels, kernel_size, batch_norm=True, resnet=True, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else (kernel_size // 2, kernel_size // 2 - 1)
        self.batch_norm = batch_norm
        self.resnet = resnet
        self.dropout = dropout
        layers = []
        layers.append(NoShareConv2d(input_size, hidden_channels, hidden_channels, kernel_size, padding=self.padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        layers.append(NoShareConv2d(input_size, hidden_channels, hidden_channels, kernel_size, padding=self.padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        layers.append(NoShareConv2d(input_size, hidden_channels, hidden_channels, kernel_size, padding=self.padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        self.resblock = ResNet(nn.Sequential(*layers), resnet=resnet)
    
    def forward(self, x):
        return F.leaky_relu(self.resblock(x))
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

class NoShareConvLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size1, kernel_size2, kernel_size3, batch_norm=True, pooling=True, resnet=True, dropout=0.):
        super().__init__()
        self.input_size1 = 32
        self.input_size2 = 16 if pooling else 32
        self.input_size3 = 8 if pooling else 32
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.padding1 = kernel_size1 // 2
        self.padding3 = kernel_size3 // 2
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.resnet = resnet
        self.dropout = dropout
        layers = []
        layers.append(NoShareConv2d(self.input_size1, in_channels, hidden_channels, self.kernel_size1, padding=self.padding1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LeakyReLU())
        if pooling:
            layers.append(nn.MaxPool2d(2))
        layers.append(NoShareResBlock(self.input_size2, hidden_channels, self.kernel_size2, batch_norm, resnet, dropout))
        if pooling:
            layers.append(nn.MaxPool2d(2))
        layers.append(NoShareConv2d(self.input_size3, hidden_channels, out_channels, self.kernel_size3, padding=self.padding3))
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
    
    
class NoShareConvNet(nn.Module):
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
        self.convlayer = NoShareConvLayer(in_channels, hidden_channels, out_channels, kernel_size1, kernel_size2, kernel_size3, batch_norm, pooling, resnet, convdropout)
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
        
        