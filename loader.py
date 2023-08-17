import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import os, shutil
import numpy as np
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings(action='ignore')

import cv2


def list_dir(path):
    return [file for file in os.listdir(path) if not file.startswith('.')]

def check_mkdir(dir_name, is_clear=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if is_clear:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)

def get_data(path, img_size=(256, 256)):
    """
        Load images and masks from path
        Args:
            path -> str
            img_size -> tuple
        Returns:
            X, y -> np.ndarray, np.ndarray
    """
    imgs_path = os.path.join(path, 'images')
    masks_path = os.path.join(path, 'masks')

    imgs_ids = sorted(list_dir(imgs_path))
    masks_ids = sorted(list_dir(masks_path))

    H, W = img_size

    X = np.zeros((len(imgs_ids), H, W, 3), dtype=np.float32)
    y = np.zeros((len(masks_ids), H, W, 1), dtype=np.float32)

    for i in range(len(imgs_ids)):
        img = cv2.imread(os.path.join(imgs_path, imgs_ids[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        mask = cv2.imread(os.path.join(masks_path, masks_ids[i]), -1)
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=-1)

        X[i] = np.asarray(img) / 255
        y[i] = np.asarray(mask).astype(bool)*255 / 255

    X = np.transpose(X, (0, 3, 1, 2))
    y = np.transpose(y, (0, 3, 1, 2))

    return X, y

class ISIC(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def Onehot(mask, n_cls=2):
    '''
    Convert mask to onehot encoding
        0 -> background
        1 -> object
    '''
    mask = mask.squeeze(1)
    onehot = F.one_hot(mask.long(), n_cls).permute(0, 3, 1, 2).float()
    return onehot

if __name__ == '__main__':
    # path = '/Users/dawn/Desktop/GSAL/data/ISIC2016_Segmentation'
    # X, y = get_data(path)
    # print(X.shape)
    # print(y.shape)
    test = torch.ones((1, 1, 4, 4))
    oh = Onehot(test)
    print(oh.shape)