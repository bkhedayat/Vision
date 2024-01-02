from torch.utils.data import Dataset
from data_utils import read_main_data, prepare_paths
from Utils.utils import parse_yaml

import sys
from pathlib import Path

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

class Data:
    """
    dataset wrapper recieves data yaml file
    """
    def __init__(self, data_yaml, split_ratio=0.8):
        self.hyper = parse_yaml(data_yaml)
        self.class_names = self.hyper['class_names']
        self.main_data_dir = str((ROOT/"data/dataset/main/").resolve())
        self.images = []
        self.annotations = []
        self.split_ratio = split_ratio
    
    def split_data(self, index_list):
        """
        split data using the split ratio and copy img + annot to 
        correspondig directories
        : param index_list: list of shuffled indexes of all data
        """
        # calculcate number of index for tarin data
        num_train_data = int(len(index_list)*self.split_ratio)

        # calculate number of index for valid data
        num_valid_data = int(len(index_list)*(1-self.split_ratio)/2)

        # split the images and annotation based on the train split
        for num, idx in enumerate(index_list):
            # create the absolute path for image and annotation files
            img_path = self.main_data_dir + "/" + self.images[idx]
            annot_path = self.main_data_dir + "/" + self.annotations[idx]
            
            # copy the files
            if num < num_train_data:
                # train dir"
                shutil.copy(img_path, str(ROOT/"data/dataset/train/img"))
                shutil.copy(annot_path, str(ROOT/"data/dataset/train/label"))
                print("create_datasets: LOG: train data index {} added!".format(idx))
            elif num >= num_train_data and num <= (num_train_data + num_valid_data):
                # valid dir"
                shutil.copy(img_path, str(ROOT/"data/dataset/valid/img"))
                shutil.copy(annot_path, str(ROOT/"data/dataset/valid/label"))
                print("create_datasets: LOG: valid data index {} added!".format(idx))
            else:
                # test dir
                # train dir"
                shutil.copy(img_path, str(ROOT/"data/dataset/test/img"))
                shutil.copy(annot_path, str(ROOT/"data/dataset/test/label"))
                print("create_datasets: LOG: test data index {} added!".format(idx))

    def create_datasets(self, cls_num):
        """
        Creates train, valid and test sets.

        Args:
            cls_num (int): number of classes to detect

        Returns:
            train, valid, test datasets (list)

        """

        # check the class number with len of class names
        assert len(self.class_names) == cls_num, "create_datasets: ERROR: class names & number of classes are different!"

        # resolve the paths for train, val and test and make them absolute
        for x in "train_dir", "valid_dir", "test_dir":
            if self.hyper.get(x):
                if isinstance(self.hyper[x], str):
                    # resolve the path and store the string
                    self.hyper[x] = str((ROOT/self.hyper[x]).resolve())

                    # prepare sub directores for the given path
                    prepare_paths(self.hyper[x])

            else:
                print("create_datasets: ERROR: {} doest not exist in yaml!".format(x))
                # exit the program
                sys.exit(1)

        
        # get list of dataset indexs, images and annotations
        idxs, image_list, annot_list = read_main_data()
        
        # copy the train / valid / test data in corresponding dir
        self.split_data(idxs)          

        # return dataset dictionary   
        return self.hyper

if __name__ == "__main__":
    #hyper = str(ROOT/"data/data_aug.yaml")
    data = Data(hyper, 0.8)
    data.create_datasets(1)