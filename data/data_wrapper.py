from torch.utils.data import Dataset
import cv2 as cv

from data_utils import read_main_data, prepare_paths, copy_data
from Utils.utils import parse_yaml

import sys
from pathlib import Path

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

class CustomDataset(Dataset):
    """
    A Map-style custom dataset

    Augs:
        image_paths (list): list of image paths
        label_paths (list): list of label paths
        transform (obj): data augmentation pipeline
    """
    def __init__(self, image_paths, label_paths, transform=False):
        super().__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        """
        Returns the length of the dataset

        Returns:
            int: number of samples in dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Returns the image and annotation at the given index

        Args:
            index (int): desired index
        """
        # get the image path & label  
        image_path = self.image_paths[index]
        label = self.label_paths[index]

        # read the image
        image = cv.imread(image_path)

        # change the channel format BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # transform the image using augmentation pipline
        if self.transfrom is not None:
            image = self.transform(image=image)["image"]

        return image, label


class DatasetHelper:
    """
    DatasetHelper prepares the main dataset and directories
    required for training and testing using data_aug.yaml

    Args:
        data_yaml (str): path to data_aug.yaml
        split_ratio (float): train-valid-test split ratio
    """
    def __init__(self, data_yaml, split_ratio=0.8):
        self.hyper = parse_yaml(data_yaml)
        self.class_names = self.hyper['class_names']
        self.main_data_dir = str((ROOT/"data/dataset/main/").resolve())
        self.split_ratio = split_ratio
    
    def split_data(self, image_list, label_list):
        """
        Split data using the split ratio and copy img + annot to directories
        
        Args:
            image_list (list): list of images
            label_list (list): list of labels
        
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

    def read_main_data(self, main_dir):
        """
        Reads the images and labels from main data directory
    
        Args:
            main_dir(str): path of the main data directory
    
        Returns: 
            images(list): list of images
            labels(list): list of labels
        """
        images = []
        labels = []
        # get the images and labels from main dataset
        for root, dirs, files in os.walk(main_dir):
            for file in files:
                if file.endswith(".png"):
                    # append the file to the list 
                    images.append(file)
                else:
                    labels.append(file)

        return images, labels

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
        image_list, label_list = self.read_main_data(self.main_data_dir)
        
        # copy the train / valid / test data in corresponding dir
        self.split_data(image_list, label_list)          

        # return dataset dictionary   
        return self.hyper