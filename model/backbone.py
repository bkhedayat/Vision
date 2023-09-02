import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    # class ConvLayer constructs a conv layer, BatchNorm and SiLU constructed with args:
    # [layer_name, input_channel, output_channel, kernel, stride, padding, groups, dilation]
    def __init__(self, layer_name, c_in, c_out, k=1, s=1, p=None, g=1, d=1, activation=True):
        super().__init__()
        self.index = layer_name
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.SiLU() if activation is True else nn.Identity()
    
    def forward(self, x):
        # apply CBS block input and return the output
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
    def get_index(self):
        print(self.index)

class Bottleneck(nn.Module):
    # class Bottleneck implements the block to reduce the number of channels in feature map using args:
    # [input_channel, output_channel, groups, c_hidden_ratio]
    def __init__(self, c_in, c_out, g=1, e=0.5):
        super().__init__()
        c_hidden = int(e * c_out)
        self.conv1 = ConvLayer("bottleneck_1", c_in, c_hidden, 1, 1)
        self.conv2 = ConvLayer("bottleneck_2", c_hidden, c_out, 3 ,1, g=g)

    def forward(self, x):
        return self.conv2(self.conv1(x))