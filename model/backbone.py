import torch
import torch.nn as nn
import torch.nn.functional as F


class CBS_block(nn.Module):
    # class CBS to provide Conv layer, BatchNorm and SiLU 
    def __init__(self, input_channel, output_channel, stride, num):
        super().__init__()
        self.index = num
        self.conv = nn.Conv2d(input_channel, out_channels=output_channel, stride=stride)
        self.bn = nn.BatchNorm2d(output_channel)
    
    def create(self, x):
        # apply CBS block input and return the output
        x = self.conv(x)
        x = self.bn(x)
        x = F.silu(x)
        return x

