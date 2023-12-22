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
    if os.path.exists(str(dir)):
        shutil.rmtree(str(dir))
            
    # create new sub dirs for images and labels
    os.mkdir(str(dir))
    os.mkdir(str(dir/"img"))
    os.mkdir(str(dir/"label"))

def read_main_data(main_dir):
    """
    reads the images and annotations from main dir
    returns index, image and annotation lists
    : param main_dir is the path to the main data directory
    """
    idxs = []
    images = []
    annotations = []
    # get the images and annotations from main dataset
    for root, dirs, files in os.walk(main_dir):
        idx = 0
        for file in files:
            if file.endswith(".png"):
                # append the file to the list 
                images.append(file)

                # append the index to index list
                idxs.append(idx)
                idx += 1
            else:
                annotations.append(file)

    # shuffle the index list
    random.shuffle(idxs)
    return idxs, images, annotations