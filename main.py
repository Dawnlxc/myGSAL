import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy
import math
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

from loss import BCEDiceLoss, AdptWeightBCEDiceLoss
from gen_net import Generator
from dsc_net import Discriminator
from config import Configure
from utils import get_data

PATH = ''

def train(img_size=(256, 256), path=PATH):
    X, y = get_data(path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


    return

if __name__ == '__main__':
    train()
