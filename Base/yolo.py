import torch.nn as nn
from Utils.utils import *

from .backbone import ConvLayer, C3Layer, SPPF
from .neck import Concat
from .head import Detect

class LayerData:
    """ Holds the parsed data of the layers"""
    def __init__(self, index, input_layer, type, args):
        self.index = index              # index of the layer
        self.input_layer = input_layer  # input from previous layer or other layers defined by index    
        self.type = type                # type of layer: ConvLayer, C3Layer, Detect, etc.
        self.args = args                # list of arguments that is essential to create a layer

    def create_module(self, amount, num_classes, anchors) -> None:
        """ Creates a module from the layer. """  
        try:
            # create a module from layers  
            if amount > 1:
                layers = []
                for _ in range(amount):
                    layers.append(self.type(*self.args))
                module = nn.Sequential(*(layers))
            else:
                if(self.type == Detect):
                    self.args.append(num_classes)   # append the num of classes
                    self.args.append(anchors)       # append the anchors
                module = self.type(*self.args)
                
            # calculate each number of parameters for each module
            module.num_params = sum(x.numel() for x in module.parameters())
                
            # assign module's index and input_from_layers
            module.index, module.input_from_layers = self.index, self.input_layer
        
            LOGGER.info(f"create_module: layer{self.index}, type:{self.type}, num_layers:{amount}, args:{self.args} added.")
            return module
        except Exception as exp:
            LOGGER.error(f"create_module: failed: {type(exp)}: {exp}")
            raise Exception(f"could not create module from layer{self.index}: {self.type}")

class Model(nn.Module):
    """ Create Model using layers of backbone, neck amd head from config.yaml file. """ 
    def __init__(self, config, channel_num=3):
        super().__init__()
        self.config = parse_yaml(config)    # path to config.yaml
        self.ch_input_list= [channel_num]   # ch_input_list contais input channel for each layer. First element: channel_num
        self.layers = []                    # list of layers

    def parse_model(self):
        """ Get different components from the config.yaml and construct the model. """  
        try:
            # get the yaml parameters
            num_classes = int(self.config["num_classes"])
            anchors = self.config['anchors']
            backbone = self.config["backbone"]
            neck = self.config["neck"]
            head = self.config["head"]

            for iter, (input_from_layers, num_layers, layer_type, args) in enumerate( backbone + neck + head):
                # use eval to change str to class
                layer_type = eval(layer_type)

                if layer_type in {ConvLayer, C3Layer, SPPF}:
                    # assign the size of input and output channels       
                    ch_in, ch_out = self.ch_input_list[input_from_layers], args[0]
                    self.ch_input_list.append(ch_out)

                    # reformat args list
                    args = [ch_in, ch_out, *args[1:]]
                
                    # C3 layers has more than 1 layer
                    if layer_type is C3Layer:
                        args.insert(2, num_layers)

                elif layer_type is Concat:
                    ch_out = sum(self.ch_input_list[layer] for layer in input_from_layers)
                    self.ch_input_list.append(ch_out)

                elif layer_type is nn.Upsample:
                    self.ch_input_list.append(self.ch_input_list[-1])

                elif layer_type is Detect:
                    args.append([self.ch_input_list[layer] for layer in input_from_layers])
                    self.ch_input_list.append(ch_out)

                else:
                    raise Exception(f"parse model: unknow layer type: {layer_type}")

                # create LayerData object
                layer = LayerData(iter, input_from_layers, layer_type, args)
            
                # add module to the layer list
                self.layers.append(layer.create_module(num_layers, num_classes, anchors))
            
            LOGGER.info("parse_model: model created!")
            return nn.Sequential(*self.layers)
        except Exception as exp:
            LOGGER.error(f"parse_model: model NOT parsed, {type(exp)}, {exp}")
            raise Exception("parse_model: could not create model.")
        
    

        
