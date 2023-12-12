import sys
import torch.nn as nn
from ..Utils.utils import parse_yaml

sys.path.append("./Base")

from backbone import ConvLayer, C3Layer, SPPF
from neck import Concat
from head import Detect


class Model(nn.Module):
    """
    class model get yaml config file and parse it. Creates backbone, neck and head components and fuses them
    ch_number is the number of channels of the input images
    class_number defines how many classes model can detect
    config is the path to model config yaml file
    """ 
    def __init__(self, config, ch_num=3, class_num=1):
        super().__init__()
        self.config = config
        self.ch_num = ch_num
        self.class_num = class_num

    def parse_model(self):
        """
        get different components from the config.yaml and construct the model
        """
        # get different components from the model dictionary
        mode_config = parse_yaml(self.config)
        
        # get number of classes
        num_classes = mode_config["num_classes"]
        print("Number of classes: {}".format(num_classes))

        # get the anchor boxes
        anchors = mode_config['anchors']
        print("Anchor boxes: {}".format(anchors))
        
        print("Parsing backbone, neck and head data:")
        layers = []
        for iter, (input_from, num_layers, layer_type, args) in enumerate(mode_config["backbone"] + mode_config["neck"] + mode_config["head"]):
            # create args list
            for indx,value in enumerate(args):
                args[indx] = eval(value) if isinstance(value, str) else value

            # edit args lits based on the layer type 
            layer_type = eval(layer_type)

            # backbone and neck layers
            if layer_type in {ConvLayer, C3Layer, SPPF}:
                # assign the size of input and output channels       
                c_in, c_out = self.ch_num[input_from], args[0]
                args = [c_in, c_out, *args[1:]]
                
                # C3 layers has more than 1 layer
                if layer_type is C3Layer:
                    args.insert(2, num_layers)
            
            # neck Concat layer 
            elif layer_type is Concat:
                c_out = sum(self.ch_num[layer] for layer in input_from) 

            # head Detection layer
            elif layer_type is Detect:
                args.append([self.ch_num[layer] for layer in input_from])    

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
