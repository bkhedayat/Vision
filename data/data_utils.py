import os, shutil, random

def create_data_dir(dir) -> None:
    """ Creates a directories for img data and label. """
    # check if the train/val/test dirs are available and delete them
    if os.path.exists(str(dir)):
        shutil.rmtree(str(dir))
            
    # create new sub dirs for images and labels
    os.mkdir(str(dir))
    os.mkdir(str(dir/"img"))
    os.mkdir(str(dir/"label"))

def copy_data(img_path, annot_path, copy_dir, data_type):
    """
    Copy image and annotation to specified directories

    Args:
        img_path (str): absolute image path
        annot_path (str): absolute annotation path
        copy_dir (str): directory to copy image and annotations
        data_type (str): type of image data that is copied
    
    Returns:

    """
    # define directories
    img_dir = copy_dir + "/" + data_type + "/img"
    annot_dir = copy_dir + "/" + data_type + "/label"

    # copy files to directories
    shutil.copy(img_path, img_dir)
    shutil.copy(annot_path, annot_dir)
    
    print("copy_data: LOG: {} data added!".format(data_type))