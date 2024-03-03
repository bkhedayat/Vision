import torch
import torch.nn as nn

from Utils.utils import LOGGER

class Concat(nn.Module):
    """ Concatinates list of tensors along the given dimension. """
    def __init__(self, dimention=1):
        super().__init__()
        self.dimension = dimention

    def forward(self, x):
        try:
            return torch.cat(x, self.dimension)
        except Exception as exp:
            LOGGER.error(f"Concat-forward: error: {type(exp)}: {exp}")
            raise Exception("Concat-forward: failed!")