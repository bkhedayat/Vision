import sys
import yaml
import torch
import torch.nn as nn
from copy import deepcopy

sys.path.append("./Base")

from backbone import ConvLayer, C3Layer, SPPF

class YamlParser:
    # YamlParser parses yaml config file and create the model
    def __init__ (self, file_name):
        self.file_name = file_name
        with open(self.file_name) as f:
            # parse yaml file safely for untrusted input
            self.model_dic = yaml.safe_load(f)

    def parse_config(self, ch):
        # get different components from the model dictionary
        dic = deepcopy(self.model_dic)
        
        # get number of classes
        num_classes = dic["num_classes"]
        print("Number of classes: {}".format(num_classes))

        # get the anchor boxes
        anchors = dic['anchors']
        print("Anchor boxes: {}".format(anchors))
        
        print("Parsing backbone and head data:")
        layers = []
        for iter, (input_from, num_layers, layer_type, args) in enumerate(dic["backbone"]):
            # create args list
            for indx,value in enumerate(args):
                args[indx] = eval(value) if isinstance(value, str) else value

            # edit args lits based on the layer type 
            layer_type = eval(layer_type)
            if layer_type in {ConvLayer, C3Layer, SPPF}:
                # assign the size of input and output channels       
                c_in, c_out = ch[input_from], args[0]
                args = [c_in, c_out, *args[1:]]
                
                # C3 layers has more than 1 layer
                if layer_type is C3Layer:
                    args.insert(2, num_layers)

            # create a sequential module of layers    
            module = nn.Sequential(*(layer_type(*args) for _ in range(num_layers))) if num_layers > 1 else layer_type(*args)

            # edit layer type and assign it to the module                 
            module.layer_type = str(layer_type)[8:-2].replace('__main__.', '')

            # calculate model parameters
            module.num_params = sum(x.numel() for x in module.parameters())

            # index, input from
            module.index, module.input_from = iter, input_from
            
            # added module to the layer list
            layers.append(module)

            print("layer{}: type:{} with num_param:{} added to Sequential.".format(module.index, module.layer_type, module.num_params))

        return nn.Sequential(*layers)

                    
                
            


class Model(nn.Module):
    # class model get yaml config file and parse it. Creates backbone, neck and head components and fuses them
    # ch_number is the number of channels of the input images
    # class_number defines how many classes model can detect 
    def __init__(self, config="config.yaml", ch_number=3, class_number=1):
        super().__init__()
        self.parser = YamlParser(config)
        self.model = self.parser.parse_config([ch_number])