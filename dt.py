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
        skeleton = torch.zeros_like(masks)
        boundary = torch.zeros_like(masks)
        for i in range(len(masks)):
            dis_map = torch.tensor(distance(masks[i].cpu()), dtype=torch.float32)
            # Normalize
            skeleton[i] = dis_map / torch.max(dis_map)
            boundary[i] = masks[i] - skeleton[i]
            # plt.imshow(skeleton[i].detach().numpy()[0, :, :], cmap='gray')
            # plt.show()
        skeleton = skeleton.detach()
        boundary = boundary.detach()
        return skeleton, boundary


# if __name__ == '__main__':
#     path = '/Users/dawn/Desktop/GSAL/data/ISIC2016_Segmentation/masks/ISIC_0000000_Segmentation.png'
#     img = cv2.imread(path, -1)
#     # s, b = DistanceTransform()(img)
#     dis_map = np.transpose(distance(img), (2, 0, 1))
#     cv2.imshow('Distance Map', dis_map)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
