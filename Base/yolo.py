import yaml
import torch
import torch.nn as nn
from copy import deepcopy


class YamlParser:
    # YamlParser parses yaml config file and create the model
    def __init__ (self, file_name):
        self.file_name = file_name
        with open(self.file_name) as f:
            # parse yaml file safely for untrusted input
            self.model_dic = yaml.safe_load(f)

    def parse_config(self, ch):
        dic = deepcopy(self.model_dic)
        # get different components from the model dictionary
        num_classes, anchors = dic['num_classes'], dic['anchors']
        for iter, (input, num_layers, type, args) in enumerate(dic["backbone"]):
            print("Parsing backbone data:")
            print("{}, {}, {}, {}".format(input, num_layers, type, args))
        
            


class Model(nn.Module):
    # class model get yaml config file and parse it. Creates backbone, neck and head components and fuses them
    # ch_number is the number of channels of the input images
    # class_number defines how many classes model can detect 
    def __init__(self, config="config.yaml", ch_number=3, class_number=1):
        super().__init__()
        self.parser = YamlParser(config)
        self.model = self.parser.parse_config(ch_number)
