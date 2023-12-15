from pathlib import Path
import sys

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

from Utils.utils import parse_yaml

class DataWrapper():
    """
    dataset wrapper
    recieves data yaml file
    """

    def __init__(self, data_yaml):
        self.hyper = parse_yaml(data_yaml)
        self.class_names = self.hyper['class_names']
        self.train_dir = ROOT/self.hyper["train_dir"]
        self.valid_dir = ROOT/self.hyper["valid_dir"]
        self.test_dir = ROOT/self.hyper["test_dir"]

    def create_datasets(self, cls_num):
        """
        Creates train, valid and test sets
        """
        
        # check the class number with len of class names
        assert len(self.class_names) == cls_num, "create_datasets: ERROR: class names & number of classes are different!"
