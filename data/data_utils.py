import sys
import os, shutil, random
from pathlib import Path

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

def prepare_paths(dir):
    """ 
    deletes exisiting and all sub dir of the given path &
    recreate them with img and label sub dirs     
    : param dir is the path object to the directory
    """

    # check if the train/val/test dirs are available and delete them
    if os.path.exists(dir):
        shutil.rmtree(dir)
            
    # create new sub dirs for images and labels
    os.mkdir(str(dir))
    os.mkdir(str(dir/"img"))
    os.mkdir(str(dir/"label"))