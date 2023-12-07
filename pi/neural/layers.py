# Created by jing at 06.11.23

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Linear


class Conv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True, active_function="LeakyReLU"):
        # Call _ConvNd constructor
        super(Conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, (0, 0),
                                   groups, bias, padding_mode='zeros')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.active_LeakyReLU = nn.LeakyReLU(0.01)
        self.active_ReLU = nn.ReLU()
        self.active_Sigmoid = nn.Sigmoid()
        self.active_Tanh = nn.Tanh()
        self.active_name = active_function
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.bn1(self.conv(x))

        if self.active_name == "LeakyReLU":
            return self.active_LeakyReLU(x)
        elif self.active_name == "Sigmoid":
            return self.active_Sigmoid(x)
        elif self.active_name == "ReLU":
            return self.active_ReLU(x)
        elif self.active_name == "Tanh":
            return self.active_Tanh(x)
        elif self.active_name == "":
            return x
        else:
            raise ValueError




# Normalized Convolution Layer
class GConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(GConv, self).__init__(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, False, (0, 0),
                                    groups, bias, padding_mode='zeros')

        self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()

    def forward(self, x):
        # Normalized Convolution
        x_g = self.active_g(self.conv_g(x))
        x_f = self.active_f(self.conv_f(x))
        x = x_f * x_g
        return x


def conv1x1(in_planes: int, out_planes: int, stride) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)


def gconv1x1(in_planes: int, out_planes: int, stride) -> GConv:
    """1x1 convolution"""
    return GConv(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)
