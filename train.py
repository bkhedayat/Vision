import sys
import os
import time
from tqdm import tqdm
from pathlib import Path
from Base.yolo import Model
from Base.loss import Loss
from Utils.utils import check_file, parse_yaml
from Utils.logger import *
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

def train(inputs, device) -> None:
    """ Starts model's training using input arguments. """
    # define output dir and weight params
    output = ROOT/"output"
    weights_dir = output/"weights_dir"
    last_weight = weights_dir/"last.pt"
    best_weight = weights_dir/"best.pt"

    # create dirs
    output.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    try:
        # get the training hyperparameters and create model
        train_hyp = parse_yaml(inputs.hyp)
        model = create_yolo_model()
        # load model on device and get the model parameters
        model.to(device)
        model_params = list(param for param in model.parameters() if param.requires_grad)
        LOGGER.error(f"train: model created and parameters saved.")
    except RuntimeError as err:
        LOGGER.error(f"train: exception catched. {err}")

    # layers to be frozen
    frozen_layers = [f"model.{layer}" for layer in (inputs.freeze if len(inputs.freeze) > 1 else range(inputs.freeze[0]))]

    try:
        # prepare datasets create dataloader from generated datasets
        train_dataloader, valid_dataloader, test_dataloader = prepare_data(inputs.data)
        LOGGER.error(f"train: train/valid dataloader created.")
    except RuntimeError as err:
        LOGGER.error(f"train: train/valid dataloader NOT created. {err}")

    # define optimzer, scheduler, loss calculator, grad scalar and num_batches
    optimizer = torch.optim.SGD(model_params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
    loss_calculator = Loss(model)
    grad_scaler = torch.cuda.amp.GradScaler(True)
    num_batches = len(train_dataloader) 

    # train model
    epochs = 10
    start = time.time()
    LOGGER.info("train: training started.")
    for epoch in range(epochs):
        try:
            model.train()
            progress_bar = enumerate(train_dataloader)
            progress_bar = tqdm(progress_bar, total=num_batches)
            optimizer.zero_grad()
            for idx, images, labels in progress_bar:
                # normalize the image with type float32
                images = images.to(device).float() / 255

                # Forward
                with torch.cuda.amp.autocast(True):
                    # predict 
                    prediction = model(images)

                    # calcualte loss
                    loss = loss_calculator(prediction, labels.to(device))

            # Backward
            grad_scaler.scale(loss).backward()    
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # update the scheduler
            scheduler.step()
            LOGGER.info(f"train: epoch_{epoch} finished. loos: {loss}")
        except RuntimeError as err:
            end = time.time()
            LOGGER.error(f"train: training stopped at epoch_{epoch}, Elapsed time {end - start}s")

    end = time.time()
    LOGGER.info(f"train: training finished. Elapsed time: {end - start}s")

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

    
