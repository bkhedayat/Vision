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

