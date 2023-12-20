from pathlib import Path
import sys
import os, shutil, random

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

from Utils.utils import parse_yaml

class Data:
    """
    dataset wrapper recieves data yaml file
    """
    def __init__(self, data_yaml):
        self.hyper = parse_yaml(data_yaml)
        self.class_names = self.hyper['class_names']
        self.main_data_dir = str((ROOT/"data/dataset/main").resolve())


    def prepare(self, dir):
        """
        create a directory for train / val / test and their sub-directories
        : param dir: path to directory
        """

        # check if the train/val/test dirs are available and delete them
        if os.path.exists(dir):
            shutil.rmtree(dir)
            
        # create new sub dirs for images and labels
        os.mkdir(dir)
        os.mkdir(dir + "\\img")
        os.mkdir(dir + "\\label")
        

    def create_datasets(self, cls_num):
        """
        Creates train, valid and test sets
        """

        # check the class number with len of class names
        assert len(self.class_names) == cls_num, "create_datasets: ERROR: class names & number of classes are different!"

        # resolve the paths for train, val and test and make them absolute
        for x in "train_dir", "valid_dir", "test_dir":
            if self.hyper.get(x):
                if isinstance(self.hyper[x], str):
                    # resolve the path and store the string
                    self.hyper[x] = str((ROOT/self.hyper[x]).resolve())
                    print(self.hyper[x])
                    # prepare dir
                    self.prepare(self.hyper[x])

            else:
                print("create_datasets: ERROR: {} doest not exist in yaml!".format(x))
                # exit the program
                sys.exit(1)

        

        # return dataset dictionary   
        return self.hyper

if __name__ == "__main__":
    hyper = str(ROOT/"data/data_aug.yaml")
    data = Data(hyper)
    data.create_datasets(1)