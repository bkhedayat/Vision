from typing import Any
import torch 
import torch.nn as nn

class FocalLoss(nn.Module):
    """ Class FocalLoss implements the focal loss function wrapped around BCEwithLogits loss fucntion"""
    def __init__(self, loss_func, gamma=1.5, alpha=0.25):
        super.__init__()
        self.loss_func = loss_func
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_func.reduction
        self.loss_func.reduction = 'none'
    
    def forward(self, prediction, truth):
        # calculate the loss using prediction and truth labels
        loss = self.loss_func(prediction, truth)

        # calculate normalized prediction probability 
        prediction_p = torch.sigmoid(prediction)

        # adjust the prediction probability
        p_t = truth * prediction_p + (1 - truth) * (1 - prediction_p)

        # calculate the alpha factor
        alpha_factor = truth * self.alpha + (1 - truth) * (1 - self.alpha)

        # calculate the modulating factor
        modulating_factor = (1.0 - p_t) ** self.gamma

        # update loss 
        loss *= alpha_factor * modulating_factor

        # return type based on the loss function reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class Loss:
    """
    Implements loss function calculations

    Args:
        model(PyTorch Model): a defined object detection model
    
    Returns:
        loss
    """
    def __init__(self, model, autobalance=False):
        # get the detection module from the model
        detect_module = model.model[-1]

        # create class members
        self.num_anchors = detect_module.na
        self.num_classes = detect_module.nc
        self.num_layers = detect_module.nl
        self.anchors = detect_module.anchors
        self.device = next(model.parameters()).device

        # get model hyper-parameters
        hyper = model.hyp

        # BCE for class and objects as criteria
        BCE_class = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyper["cls_pw"]], device=self.device))
        BCE_object = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyper["obj_pw"]], device=self.device))

        # get the gamma from model hyper-parameters
        gamma = hyper["f1_gamma"]

        # use focal loss as the criteria
        if gamma > 0:
            BCE_class, BCE_object = FocalLoss(BCE_class, gamma), FocalLoss(BCE_object, gamma)
        
    def __call__(self, prediction, labels):
        # create loss tensors for class, box and object
        class_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        object_loss = torch.zeros(1, device=self.device)

        # extract the targets from the labels
        class_target, box_target, indices, anchors = self.get_targets(prediction, labels)

        
