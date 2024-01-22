import os
import yaml
from copy import deepcopy

def check_file(file):
    """
    checks if the file exists
    """ 
    # convert file to str format
    file = str(file)
    if os.path.isfile(file):
        return file
    else :
        return ""
    
def parse_yaml(path):
    """
    parse the yaml file using path to the file
    """
    with open(path) as f:
        # load the yaml file
        loaded = yaml.safe_load(f)
    
    # make a deep copy of the parsed yaml file
    loaded = deepcopy(loaded)
    return loaded

def Check_input_size(img_input_size):
    """
    Checks whether the input size is divisable to 32

    Args:
        img_input_size(int): the input size of the image

    Returns:
        (bool)
    """
    reminder = int(img_input_size) % 32 
    return True if reminder == 0 else False