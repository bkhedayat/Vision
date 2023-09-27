import sys
import os
from Base.yolo import Model

def AddPaths():
    # adds all the necessary paths to the system path
    model_dir = os.getcwd() + "\\Base"
    sys.path.append(model_dir)
    print("Path: {} added!".format(model_dir))

def CreateModel():
    # creates yolo model and return it
    vision_model = Model("Base/config.yaml", 3)
    return vision_model

def train():
    # starts training
    pass

def print_version(version):
    print(version)

if __name__ == "__main__":

    # add the paths
    AddPaths()

    # create the yolo model
    model = CreateModel()
