import torch
import torch.nn as nn

class Concat(nn.Module):
# class concat concatinates list of tensors along the given dimension
    def __init__(self, dimention=1):
        super().__init__()
        self.dimension = dimention

    def forward(self, x):
        return torch.cat(x, self.dimension)