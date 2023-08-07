import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

class BCEDiceLoss(nn.Module):
    '''
        L_local = BCEDice(s, s_gt) + BCEDice(b, b_gt)
        Args:
            y_pred -> Tensor; segmentation mask
            y_gt -> Tensor; ground truth mask
        Returns:
            loss -> Tensor; loss value
    '''
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, y_pred, y_gt):
        bce = F.binary_cross_entropy_with_logits(y_pred, y_gt)
        y_pred = torch.sigmoid(y_pred)
        n = y_gt.size(0)
        y_pred = y_pred.view(n, -1)
        y_gt = y_gt.view(n, -1)

        smooth = 1e-5

        intersection = (y_pred * y_gt)
        dice = (2. * intersection.sum(1) + smooth) / (y_pred.sum(1) + y_gt.sum(1) + smooth)
        dice = 1 - dice.sum() / n
        return bce + dice

class AdptWeightBCEDiceLoss(nn.Module):
    def __init__(self):
        super(AdptWeightBCEDiceLoss, self).__init__()
        self.smooth = 1e-8
        pass
    def forward(self, y_pred, y_target):
        weight = 1 + 5*torch.abs(F.avg_pool2d(y_target, kernel_size=31, stride=1, padding=15) - y_target)
        bce = F.binary_cross_entropy_with_logits(y_pred, y_target, reduce='none')
        w_bce = ((weight * bce).sum(dim=(2, 3)) + self.smooth) / (weight.sum(dim=(2, 3)) + self.smooth)

        y_pred = torch.sigmoid(y_pred)

        intersection = ((y_pred * y_target) * weight).sum(dim=(2, 3))
        union = ((y_pred + y_target) * weight).sum(dim=(2, 3))
        w_iou = 1.0 - (intersection + 1 + self.smooth) / (union - intersection + 1 + self.smooth)

        mpred = y_pred.mean(dim=(2, 3))

        return (w_bce + w_iou).mean()
