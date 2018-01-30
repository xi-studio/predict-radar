from __future__ import print_function
import argparse
import os
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from scipy import misc


def convNd(_input, batch, channel):
    f = nn.Conv1d(channel, channel, 3, padding=1, bias=False) 

    shape = _input.data.size()
    len_data = np.prod(shape)/batch/channel

    x = _input.resize(batch,channel,len_data)
    x = f(x)
    x = x.resize_as(_input)

    for num in range(len(shape)-3):
        dim = num + 3
        xi = x.transpose(2,dim)
        x = xi.resize(batch,channel,len_data)
        x = f(x)
        x = x.resize_as(xi)
        x = x.transpose(2,dim)

    output = x 
    print(output)


if __name__ == '__main__':
    x = torch.Tensor(np.ones((10,2,2,2,3,4,10)))
    x = Variable(x)
    convNd(x,10,2)    
    
    




