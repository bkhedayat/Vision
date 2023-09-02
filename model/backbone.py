import torch
import torch.nn as nn
import torch.nn.functional as F

class CBS_block(nn.Module):
    # class CBS to provide Conv layer, BatchNorm and SiLU constructed with args:
    # [layer_num, c_in, c_out, kernel, stride, padding, groups, dilation]
    def __init__(self, num, c_in, c_out, k=1, s=1, p=None, g=1, d=1, activation=True):
        super().__init__()
        self.index = num
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, d, g)
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.SiLU() if activation is True else nn.Identity()
    
    def forward(self, x):
        # apply CBS block input and return the output
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
