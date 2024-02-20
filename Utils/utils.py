import sys
import os
import yaml
from logger import *
from copy import deepcopy

def check_file(file) -> None:
    """ Checks if the file exists. """ 
    try:
        # convert file to str format
        file = str(file)
        if os.path.isfile(file) == False:
            raise Exception(f"check_file: {file} does not exist.")
        LOGGER.info(f"check_file: {file} exists.")
    except Exception as exp:
        LOGGER.error(f"check_file: {type(exp)}, {exp}")
        sys.exit(1)
    
def parse_yaml(path):
    """ Parse the yaml file using path to the file. """
    try:
        with open(path) as f:
            # load the yaml file
            loaded = yaml.safe_load(f)
            LOGGER.info(f"parse_yaml: yaml file at {path} loaded.")
            # make a deep copy of the parsed yaml file
            return deepcopy(loaded)
    except Exception as exp:
        LOGGER.error(f"parse_yaml: {type(exp)}, {exp}")
        sys.exit(1)

def Check_input_size(img_input_size) -> bool:
    """ Checks whether the input size is divisable to 32. """
    try:
        reminder = int(img_input_size) % 32 
        return True if reminder == 0 else False
    except Exception as exp:
        LOGGER.error(f"Check_input_size: {type(exp)}, {exp}")
        sys.exit(1)