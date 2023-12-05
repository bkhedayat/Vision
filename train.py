import sys
import os
from pathlib import Path
from Base.yolo import Model

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

def init():
    sw_version = 0.1
    print_version(sw_version)

    # parse arguments
    args = parse_args()

def parse_args():
    # parse the args from CLI
    parser = argparse.ArgumentParser() 
    parser.add_argument("--weights", type=str, default=ROOT/"yolov5.pt", help="path of weights")
    parser.add_argument("--config", type= str, default=ROOT/"Base/config.yaml.", help="path of config yaml")
    parser.add_argument("--data", type=str, default=ROOT/"data", help="path of data yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="number of trainning iterations")
    parser.add_argument("--batch-size", type=int default=8, help="input batch size")
    parser.add_argument("--input-size", type=int, default=512, help="input size of image in pixels")
    return parser.parse_args()

def create_yolo_model():
    # creates yolo model and return it
    vision_model = Model("Base/config.yaml", 3)
    return vision_model
s
def train():
    # starts training
    pass

def print_version(version):
    print("\n ###############################################\n")
    print("         JARVIS, Software version: {}".format(version))
    print("\n ###############################################\n")

if __name__ == "__main__":

    # initialize the software
    init()

    # create the yolo model
    model = create_yolo_model()
