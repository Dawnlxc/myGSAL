import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

import cv2
from scipy.ndimage import distance_transform_edt as distance

class DistanceTransform(nn.Module):
    def __init__(self):
        super(DistanceTransform, self).__init__()
        pass
    def forward(self, mask):
        return distance(mask)

if __name__ == '__main__':
    path = './data/1.png'
    test = torch.zeros((32, 256, 256, 3))
    distance = DistanceTransform()(test)
    print(distance)
