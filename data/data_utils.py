import os, shutil
from Utils.logger import LOGGER

def create_dir(dir) -> None:
    """ Creates a directories for img data and label. """
    # check if the train/val/test dirs are available and delete them
    if os.path.exists(str(dir)):
        shutil.rmtree(str(dir))
        LOGGER.info(f"create_dir: {dir} deleted!")
        
    # create new sub dirs for images and labels
    os.mkdir(str(dir))
    os.mkdir(str(dir/"img"))
    os.mkdir(str(dir/"label"))
    LOGGER.info(f"create_dir: {dir} created!")

def copy_data(img_path, annot_path, copy_dir, data_type) -> None:
    """ Copies image and annotation to specified directories. """
    # define directories
    img_dir = copy_dir + "/" + data_type + "/img"
    annot_dir = copy_dir + "/" + data_type + "/label"

    # copy files to directories
    shutil.copy(img_path, img_dir)
    shutil.copy(annot_path, annot_dir)
    LOGGER.info(f"copy_data: {data_type} data added!")