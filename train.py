import sys
import os
from pathlib import Path
from Base.yolo import Model
from Utils.utils import check_file
import torch

import argparse

# create a relative path from existing root directory
# get the path of the train.py
TRAIN = Path(__file__).resolve()

# get the first parrent dir as ROOT
ROOT = TRAIN.parents[0]

# add the ROOT to sys.path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# create a relative of the ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def train(inputs, device):
    """
    trains the model 
    """
    # define output dir and create it
    output = ROOT/"output"
    output.mkdir(parents=True, exist_ok=True)

    # define weights dir and create it
    weights_dir = output/"weights_dir"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # define best and last weights
    last = weights_dir/"last.pt"
    best = weights_dir/"best.pt"


    pass

def init():
    # log intro
    sw_version = 0.1
    print_version(sw_version)

    # parse arguments
    inputs = parse_args()

    # get the parameters from the inputs and check if they exist
    inputs.data, inputs.config, input.weights = check_file(inputs.data), check_file(inputs.config), str(inputs.weights)
    assert len(inputs.data), "init: ERROR: data.yaml is missing"
    assert len(inputs.config), "init: ERROR: config.yaml is missing"
    assert len(inputs.weights), "init: ERROR: model weights 'XXX.pt' is missing"

    # get the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start training
    train(inputs, device)

def parse_args():
    # parse the args from CLI
    parser = argparse.ArgumentParser() 
    parser.add_argument("--weights", type=str, default=ROOT/"yolov5.pt", help="path of weights")
    parser.add_argument("--config", type= str, default=ROOT/"Base/config.yaml.", help="path of config yaml")
    parser.add_argument("--data", type=str, default=ROOT/"data", help="path of data yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="number of trainning iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="input batch size")
    parser.add_argument("--input-size", type=int, default=512, help="input size of image in pixels")
    return parser.parse_args()

def create_yolo_model():
    # creates yolo model and return it
    vision_model = Model("Base/config.yaml", 3)
    return vision_model

def print_version(version):
    print("\n ###############################################\n")
    print("         JARVIS, Software version: {}".format(version))
    print("\n ###############################################\n")

if __name__ == "__main__":

    # initialize the software
    init()

    # create the yolo model
    model = create_yolo_model()
