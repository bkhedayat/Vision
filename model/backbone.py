import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    # class ConvLayer constructs a conv layer, BatchNorm and SiLU constructed with args:
    # [layer_num, c_in, c_out, kernel, stride, padding, groups, dilation]
    def __init__(self, num, c_in, c_out, k=1, s=1, p=None, g=1, d=1, activation=True):
        super().__init__()
        self.index = num
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.SiLU() if activation is True else nn.Identity()
    
    def forward(self, x):
        # apply CBS block input and return the output
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

