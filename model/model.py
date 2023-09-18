import yaml
import torch
import torch.nn as nn
from copy import deepcopy


class YamlParser:
    # YamlParser parses yaml config file and create the model
    def __init__ (self, file_name):
        self.file_name = file_name
    
    def parse_config(self):
        with open(self.file_name) as f:
            # parse yaml file safely for untrusted input
            self.model_config = yaml.safe_load(f)


class Model(nn.Module):
    # class model get yaml config file and parse it. Creates backbone, neck and head components and fuses them
    # ch_number is the number of channels of the input images
    # class_number defines how many classes model can detect 
    def __init__(self, config="config.yaml", ch_number=3, class_number=1):
        super().__init__()
        self.parser = YamlParser(deepcopy(config))
        self.model = self.parser.parse_config()
