import sys
import os
from Base.yolo import Model

def init():
    sw_version = 0.1
    print_version(sw_version)
    
    # add paths to systems
    add_paths()

def add_paths():
    # adds all the necessary paths to the system path
    model_dir = os.getcwd() + "\\Base"
    sys.path.append(model_dir)
    print("Path: {} added!".format(model_dir))

def create_yolo_model():
    # creates yolo model and return it
    vision_model = Model("Base/config.yaml", 3)
    return vision_model

def train():
    # starts training
    pass

def print_version(version):
    print("\n ###############################################\n")
    print("         TAKAM, Software version: {}".format(version))
    print("\n ###############################################\n")

if __name__ == "__main__":

    # initialize the software
    init()

    # create the yolo model
    model = create_yolo_model()
