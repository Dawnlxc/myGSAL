import matplotlib.pyplot as plt
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

    def forward(self, masks):
        masks = masks / np.max(masks)
        dis_map = distance(masks)
        # Normalize
        skeleton = dis_map / np.max(dis_map)
        boundary = masks - skeleton
        return skeleton, boundary

# if __name__ == '__main__':
#     path = '/Users/dawn/Desktop/GSAL/data/ISIC2016_Segmentation/masks/ISIC_0000000_Segmentation.png'
#     img = cv2.imread(path, -1)
#     img = np.expand_dims(img, axis=-1)
#
#     s, b = DistanceTransform()(img)
#     cv2.imshow('Distance Map', b)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
