import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import copy
import math
import numpy as np
import time

import warnings
warnings.filterwarnings(action='ignore')

from loss import BCEDiceLoss, AdptWeightBCEDiceLoss
from gen_net import Generator
from dsc_net import Discriminator
from config import Configure, Arguments
from dt import DistanceTransform
from utils import get_data

PATH = ''

def train(args):
    X, y = get_data(args.path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    real_label = 1.
    fake_label = 0.

    generator = Generator(in_channels=3, out_channels=1)
    skeletion_dsc = Discriminator(in_channels=1, out_channels=1)
    boundary_dsc = Discriminator(in_channels=1, out_channels=1)

    generator_optim = torch.optim.Adam(generator.parameters(), lr=1e-4)
    sdsc_optim = torch.optim.Adam(skeletion_dsc.parameters(), lr=1e-4)
    bdsc_optim = torch.optim.Adam(boundary_dsc.parameters(), lr=1e-4)

    all_seg_loss = dict()

    for n in range(args.n_epochs):

        seg_loss = 0.0
        generator.zero_grad()
        y_skeleton, y_boundary, y_seg = generator(X_train)
        seg_loss
        pass

    return

train_args = Arguments(
    path=PATH,
    img_size=(256, 256),
    n_epochs=70,
    batch_size=16,
    lr=1e-4,

)

if __name__ == '__main__':
    train()
