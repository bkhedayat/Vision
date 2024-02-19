from typing import Any
import torch 
import torch.nn as nn
from metrics import calculate_iou

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
    """ Implements loss function calculations. """
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

        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCE_class, BCE_object, 1.0, hyper, autobalance
        
    def __call__(self, prediction, labels):
        # create loss tensors for class, box and object
        class_loss = torch.zeros(1, device=self.device)
        box_loss = torch.zeros(1, device=self.device)
        object_loss = torch.zeros(1, device=self.device)

        # extract the targets from the labels
        class_target, box_target, indices, anchors = self.get_targets(prediction, labels)

        # calculate losses
        for idx, pred in enumerate(prediction):
            img, anchor, grid_y, grid_x = indices[idx]
            target_obj = torch.zeros(pred.shape[:4], dtype=pred.dtype, device=self.device)  # target obj

            # number of targets
            num_targets = img.shape[0]  
            if num_targets:
                #target-subset of predictions
                pxy, pwh, _, pcls = pred[img, anchor, grid_y, grid_x].split((2, 2, 1, self.num_classes), 1) 

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[idx]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                # calucalte IoU
                iou = calculate_iou(pbox, box_target[idx]).squeeze()  
                
                # calculate IoU loss
                box_loss += (1.0 - iou).mean()  

                # Objectness
                iou = iou.detach().clamp(0).type(target_obj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    img, anchor, grid_y, grid_x, iou = img[j], anchor[j], grid_y[j], grid_x[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                # iou ratio
                target_obj[img, anchor, grid_y, grid_x] = iou  

                # Classification loss
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(num_targets), class_target[idx]] = self.cp
                    class_loss += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pred[..., 4], target_obj)
            # object loss
            object_loss += obji * self.balance[idx]  
            if self.autobalance:
                self.balance[idx] = self.balance[idx] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # add gains to the calculated losses
        box_loss *= self.hyp["box"]
        object_loss *= self.hyp["obj"]
        class_loss *= self.hyp["cls"]

        # get the batch size
        batch_size = target_obj.shape[0]  

        return (box_loss + object_loss + class_loss) * batch_size, torch.cat((box_loss, object_loss, class_loss)).detach()

        
    def get_targets(self, predictions, targets):
        """ Creates targets to calculate loss. """
        # get number of anchors, number of targets
        num_anchors, num_targets = self.num_anchors, targets.shape[0]

        # create lists for target: class, box, indices and anchors
        idxs, anch, t_class, t_box = [], [], [], []

        # generate gain for normalization
        gain = torch.ones(7, device=self.device) 

        # anchor indexs for the targets
        anchor_idxs = torch.arange(num_anchors, device=self.device).float().view(num_anchors, 1).repeat(1, num_targets)

        # concatenate targets with anchor indexes
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), anchor_idxs[..., None], 2))

        # create an offset
        offset = (torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * 0.5)

        for idx in range(self.num_layers):
            anchors, pred_shape = self.anchors[idx], predictions[idx].shape
            # xyxy gain
            gain[2:6] = torch.tensor(pred_shape)[[3, 2, 3, 2]]  

            # Match targets to anchors
            targets_g = targets * gain  # shape(3,n,7)
            if num_targets:
                # Matches
                wh_ratio = targets_g[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.hyp["anchor_t"]  # compare
                t = targets_g[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < 0.5) & (gxy > 1)).T
                l, m = ((gxi % 1 < 0.5) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            idxs.append((b, a, gj.clamp_(0, pred_shape[2] - 1), gi.clamp_(0, pred_shape[3] - 1)))  # image, anchor, grid
            t_box.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            t_class.append(c)  # class

        return t_class, t_box, idxs, anchors