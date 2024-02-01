import sys
import os
import time
from tqdm import tqdm
from pathlib import Path
from Base.yolo import Model
from Utils.utils import check_file, parse_yaml
from data.data_wrapper import DatasetHelper, CustomDataset
import torch
from torch.optim import lr_scheduler
import argparse

# create a relative path from existing root directory
# get the path of the train.py
TRAIN = Path(__file__).resolve()

# get the first parrent dir as ROOT
ROOT = TRAIN.parents[0]

# add the ROOT to sys.path
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# create a relative of the ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def train(inputs, device):
    """
    Starts model's training using input arguments.

    Args:
        input(obj): argparser object containing input arguments

    Returns:

    """
    # define output dir and create it
    output = ROOT/"output"
    output.mkdir(parents=True, exist_ok=True)

    # define weights dir and create it
    weights_dir = output/"weights_dir"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # define best and last weights
    last = weights_dir/"last.pt"
    best = weights_dir/"best.pt"

    # get the training hyperparameters
    train_hyp = parse_yaml(inputs.hyp)

    # create the yolo model
    model = create_yolo_model()

    # layers to be frozen
    frozen_layers = [f"model.{layer}" for layer in (inputs.freeze if len(inputs.freeze) > 1 else range(inputs.freeze[0]))]

    # prepare datasets create dataloader from generated datasets
    train_dataloader, valid_dataloader, test_dataloader = prepare_data(inputs.data)

    # load model on device and get the model parameters
    model.to(device)
    model_params = list(param for param in model.parameters() if param.requires_grad)

    # define optimzer
    optimizer = torch.optim.SGD(model_params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # define scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)

    # train model
    epochs = 10
    batches = len(train_dataloader)     # number of batches 
    start = time.time()
    
    # grad scaler
    g_scaler = torch.cuda.amp.GradScaler(True)
    for epoch in range(epochs):
        
        model.train()
        progress_bar = enumerate(train_dataloader)
        progress_bar = tqdm(progress_bar, total=batches)
        optimizer.zero_grad()
        for idx, images, labels in progress_bar:
            # normalize the image with type float32
            images = images.to(device).float() / 255

            # Forward
            with torch.cuda.amp.autocast(True):
                # predict 
                prediction = model(images)

                # calcualte loss
                loss = compute_loss(prediction, labels.to(device))

        # Backward
        g_scaler.scale(loss).backward()    
        g_scaler.step(optimizer)
        g_scaler.update()

    end = time.time()

def prepare_data(data_yaml, cls_num=1):
    """
    Prepares datasets for training and validataion

    Args:
        data_yaml(str): path to the data_yaml file

    Returns:
        data loaders(obj): train, valid, test dataloaders
    """
    # create the data helper object to prepare datasets
    data_helper = DatasetHelper(data_yaml)

    # get data dictionary from data_helper
    data_dict = DatasetHelper.create_datasets(cls_num)

    # create dataloader from generated datasets
    return data_helper.create_dataloaders(data_dict=data_dict)

def init():
    # log intro
    sw_version = 0.1
    print_version(sw_version)

    # parse arguments
    inputs = parse_args()

    # get the parameters from the inputs and check if they exist
    inputs.data, inputs.config, inputs.weights, inputs.hyp = \
          check_file(inputs.data), check_file(inputs.config), str(inputs.weights), check_file(inputs.hyp)
    assert len(inputs.data), "init: ERROR: data.yaml is missing"
    assert len(inputs.config), "init: ERROR: config.yaml is missing"
    assert len(inputs.weights), "init: ERROR: model weights 'XXX.pt' is missing"
    assert len(inputs.hyp), "init: ERROR: hyp.yaml is missing"

    # get the torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start training
    train(inputs, device)

def parse_args():
    # parse the args from CLI
    parser = argparse.ArgumentParser() 
    parser.add_argument("--weights", type=str, default=ROOT/"yolov5.pt", help="path of weights")
    parser.add_argument("--config", type= str, default=ROOT/"Base/config.yaml.", help="path of config yaml")
    parser.add_argument("--data", type=str, default=ROOT/"data", help="path of data yaml file")
    parser.add_argument("--hyp", type=str, defualt=ROOT/"hyperparam.yaml", help="training hyperparameters yaml file")
    parser.add_argument("--epochs", type=int, default=100, help="number of trainning iterations")
    parser.add_argument("--batch-size", type=int, default=8, help="input batch size")
    parser.add_argument("--input-size", type=int, default=512, help="input size of image in pixels")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="freeze layers: [5] or [1, 2, 3, 4, 5]")
    return parser.parse_args()

def create_yolo_model():
    # creates yolo model and return it
    vision_model = Model("Base/config.yaml", 3)
    return vision_model

def print_version(version):
    print("\n ###############################################\n")
    print("         JARVIS, Software version: {}".format(version))
    print("\n ###############################################\n")

if __name__ == "__main__":

    # initialize the software
    init()

    
