import sys
import os

def AddPaths():
    # adds all the necessary paths to the system path
    model_dir = os.getcwd() + "\\Base"
    sys.path.append(model_dir)
    print("Path: {} added!".format(model_dir))

def train():
    # starts training
    pass

def print_version(version):
    print(version)

if __name__ == "__main__":
    # add the paths
    AddPaths()
