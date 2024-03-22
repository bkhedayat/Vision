from torch.utils.data import Dataset, DataLoader
import cv2 as cv

from .data_utils import create_dir, copy_data, LOGGER
from Utils.utils import parse_yaml

import sys
import os
from pathlib import Path

# create absolute ROOT path object of model
ROOT = Path(__file__).resolve().parents[1]

# add ROOT to sys path
sys.path.append(str(ROOT))

class CustomDataset(Dataset):
    """ A Map-style custom dataset. """
    def __init__(self, image_paths, label_paths, transform=False):
        super().__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self) -> int:
        """ Returns the length of the dataset. """
        return len(self.image_paths)

    def __getitem__(self, index) -> list:
        """ Returns the set of image and label at the given index. """
        # change the channel format BGR to RGB
        image = cv.cvtColor(cv.imread(self.image_paths[index]), cv.COLOR_BGR2RGB)
        if image == None:
            raise Exception("CustomDataset: __getitem__ could not read the data.")
        else:
            label = []
            f = open(self.label_paths[index], "r")
            label = f.readline().split(" ")

            # transform the image using augmentation pipline
            if self.transfrom is not None:
                image = self.transform(image=image)["image"]
            return [image, label]


class DatasetHelper:
    """ Prepares the main dataset and directories required for training and testing using data_aug.yaml. """
    def __init__(self, data_yaml, split_ratio=0.8) -> None:
        try:
            self.hyper = parse_yaml(data_yaml)
            self.main_data_dir = str((ROOT/"data/dataset/main/").resolve())
            self.split_ratio = split_ratio
        except Exception as exp:
            LOGGER.error(f"DatasetHelper: init error: {type(exp)}: {exp}")
            raise Exception("DatasetHelper: init failed!")

    def create_datasets(self) -> dict:
        """ Creates train, valid and test sets. """
        try:
            # resolve the paths for train, val and test and make them absolute
            for x in "train_dir", "valid_dir", "test_dir":
                if isinstance(self.hyper[x], str):
                    # create directores
                    create_dir(str((ROOT/self.hyper[x]).resolve()))
                else:
                    raise Exception(f"{x} entry is not a str type!")
                
            # get list of dataset indexs, images and annotations
            image_list, label_list = self.create_data_lists(self.main_data_dir)

            # return dataset dictionary   
            return self.split_data(image_list, label_list) 

        except Exception as exp:
            LOGGER.error(f"create_datasets: {type(exp)}: {exp}")
            raise Exception("create_datasets: failed!")
 
    def create_dataloaders(self, data_dict) -> tuple:
        """ Creates Dataset objects from data_dict lists. """
        try:
            # create Dataset objects
            train_list, valid_list, test_list = data_dict["train"], data_dict["valid"], data_dict["test"]
            train_dataset = CustomDataset(image_paths=train_list[0], label_paths=train_list[1])
            valid_dataset = CustomDataset(image_paths=valid_list[0], label_paths=valid_list[1])
            test_dataset = CustomDataset(image_paths=test_list[0], label_paths=test_list[1])
        except Exception as exp:
            LOGGER.error("create_dataloaders: create CustomDataset obj failed")
            raise Exception("create_dataloaders: failed")
        
        try:
            # create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=8, drop_last=True, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=8, drop_last=True, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1)
            return (train_loader, valid_loader, test_loader)
        except Exception as exp:
            LOGGER.error("create_dataloaders: create DataLoader obj failed")
            raise Exception("create_dataloaders: failed")

    def split_data(self, image_list, label_list) -> dict:
        """ Split data to train, dev, test using the split ratio. """
        if len(image_list) == 0 or len(label_list) == 0:
            raise Exception("split_data: empty input image or annot list!")
        num_train_data = int(len(image_list)*self.split_ratio)          # num train data
        num_valid_data = int(len(image_list)*(1-self.split_ratio)/2)    # num validation data

        # define copy directory
        copy_dir = str(ROOT/"data/dataset")
        train_image, train_label, valid_image, valid_label, test_image, test_label = []

        # split the images and annotation based on the train split
        for idx, _ in enumerate(image_list):
            # copy the files
            if idx < num_train_data:
                # train dataset"
                dataset_type = "train"
                train_image.append(image_list[idx])
                train_label.append(label_list[idx])
            elif idx >= num_train_data and idx <= (num_train_data + num_valid_data):
                # validation dataset"
                dataset_type = "valid"
                valid_image.append(image_list[idx])
                valid_label.append(label_list[idx])
            else:
                # test dataset
                dataset_type = "test"
                test_image.append(image_list[idx])
                test_label.append(label_list[idx])

            # copy the files into the specified directores
            copy_data(image_list[idx], label_list[idx], copy_dir, dataset_type)

        return {"train" : [train_image, train_label], "valid" : [valid_image, valid_label], "test" : [test_image, test_label]}
        
    def create_data_lists(self, main_dir) -> any:
        """ Reads the images and labels from main data directory and create lists of imgs and labels ."""
        images, labels = []
        # get the images and labels from main dataset
        for root, dirs, files in os.walk(main_dir):
            if len(files) == 0:
                    raise Exception("create_data_lists: main folder is empty!")
            for file in files:
                if file.endswith(".png"):
                    images.append(str(self.main_data_dir + "/" + file))
                else:
                    labels.append(str(self.main_data_dir + "/" + file))
        return images, labels