from torch.utils.data import Dataset
from data_utils import read_main_data, prepare_paths, copy_data
from Utils.utils import parse_yaml

import sys
from pathlib import Path

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

class Data:
    """
    Data class creates train/valid/test datasets from data_aug.yaml

    Args:
        data_yaml (str): path to data_aug.yaml
        split_ratio (float): train-valid-test split ratio

    Returns:
        train / valid / test atasets
    """
    def __init__(self, data_yaml, split_ratio=0.8):
        self.hyper = parse_yaml(data_yaml)
        self.class_names = self.hyper['class_names']
        self.main_data_dir = str((ROOT/"data/dataset/main/").resolve())
        self.split_ratio = split_ratio
    
    def split_data(self, index_list, image_list, annot_list):
        """
        Split data using the split ratio and copy img + annot to directories
        
        Args:
            index_list (list): shuffeld index of images
            image_list (list): list of images
            annot_list (list): list of annotations
        
        Returns:

        """
        # calculcate number of index for tarin data
        num_train_data = int(len(index_list)*self.split_ratio)

        # calculate number of index for valid data
        num_valid_data = int(len(index_list)*(1-self.split_ratio)/2)

        # define copy directory
        copy_dir = str(ROOT/"data/dataset")

        # split the images and annotation based on the train split
        for num, idx in enumerate(index_list):
            # create the absolute path for image and annotation files
            img_path = self.main_data_dir + "/" + image_list[idx]
            annot_path = self.main_data_dir + "/" + annot_list[idx]
            
            # copy the files
            if num < num_train_data:
                # train dir"
                copy_data(img_path, annot_path, copy_dir, "train")
            elif num >= num_train_data and num <= (num_train_data + num_valid_data):
                # valid dir"
                copy_data(img_path, annot_path, copy_dir, "valid")
            else:
                # test dir
                copy_data(img_path, annot_path, copy_dir, "test")


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

class CustomDataset(Dataset):
    """
    A Map-style custom dataset loader

    Augs:
        data_tensor (tuple): contains 3 tensor of image, annotations and bbox coordinates
    """
    def __init__(self, data_tesnor):
        super().__init__()
        self.data_tensor = data_tesnor

    def __getitem__(self, index):
        """
        Returns the data and annotations at the given index

        Args:
            index (int): desired index
        """
        pass
    
    def __len__(self):
        """
        Returns the length of the dataset

        Returns:
            int: number of samples in dataset
        """
        return len(self.data_tensor[0])