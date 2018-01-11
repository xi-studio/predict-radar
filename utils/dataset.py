import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random


class Radars(data.Dataset):
    def __init__(self,dataPath='facades/train',fineSize=256):
        super(Facades, self).__init__()
        # list all images into a list
        self.image_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        path = os.path.join(self.dataPath,self.image_list[index])
        img = default_loader(path) # 512x256
        #img = ToTensor(img) # 3 x 256 x 512

        # 2. seperate image A and B; Scale; Random Crop; to Tensor
        w,h = img.size
        imgA = img.crop((0, 0, w/2, h))
        imgB = img.crop((w/2, 0, w, h))

        if(h != self.loadSize):
            imgA = imgA.resize((self.loadSize, self.loadSize), Image.BILINEAR)
            imgB = imgB.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if(self.loadSize != self.fineSize):
            x1 = random.randint(0, self.loadSize - self.fineSize)
            y1 = random.randint(0, self.loadSize - self.fineSize)
            imgA = imgA.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))
            imgB = imgB.crop((x1, y1, x1 + self.fineSize, y1 + self.fineSize))

        imgA = ToTensor(imgA) # 3 x 256 x 256
        imgB = ToTensor(imgB) # 3 x 256 x 256

        imgA = imgA.mul_(2).add_(-1)
        imgB = imgB.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return imgA, imgB

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
