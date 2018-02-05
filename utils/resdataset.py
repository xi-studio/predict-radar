from __future__ import print_function
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random
import h5py
import cPickle
import gzip

def default_loader(path):
    f = h5py.File(path,'r')
    imgA = f['input'][:]
    imgB = f['output'][:]
    f.close()
    return imgA/255.0, imgB/255.0
 


class Radars(data.Dataset):
    def __init__(self,dataPath='data/idx.pkl.gz'):
        super(Radars, self).__init__()
        f = gzip.open(dataPath)
        train = cPickle.load(f)
        test = cPickle.load(f)
        f.close()
        self.image_list = train
        print(len(train))
        f.close()

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        path = self.image_list[index]
        imgA,imgB = default_loader(path) # 512x256

        # 2. seperate image A and B; Scale; Random Crop; to Tensor

        return imgA, imgB

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)


