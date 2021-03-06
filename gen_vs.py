from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model.GeneratorBN import Generator
from utils.dataset import Radars
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser(description='test pix2pix model')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loadSize', type=int, default=256, help='scale image to this size')
parser.add_argument('--fineSize', type=int, default=256, help='random crop image to this size')
parser.add_argument('--input_nc', type=int, default=3, help='channel number of input image')
parser.add_argument('--output_nc', type=int, default=3, help='channel number of output image')
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--dataPath', default='facades/test/', help='path to training images')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--imgNum', type=int, default=32, help='How many images to generate?')

def csi(a,b):
    A = a - b
    B = a + b
    
    pod = np.sum(B==2)
    far = np.sum(A>0)
    mis = np.sum(A<0)
     
    return pod,far,mis

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

###########   Load netG   ###########
assert opt.netG != '', "netG must be provided!"
netG = Generator(opt.input_nc, opt.output_nc, opt.ngf)
netG.load_state_dict(torch.load(opt.netG))
###########   Generate   ###########
dataset = Radars(dataPath=opt.dataPath)
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=2)

input_nc = opt.input_nc
fineSize = opt.fineSize

real_A = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_B = torch.FloatTensor(opt.batchSize, input_nc, fineSize, fineSize)
real_A = Variable(real_A)
real_B = Variable(real_B)

if(opt.cuda):
    netG.cuda()
    real_A = real_A.cuda()
    real_B = real_B.cuda()


lpod = np.zeros(10)
lfar = np.zeros(10)
lmis = np.zeros(10)

for i, image in enumerate(train_loader):
    print(i)
    imgA = image[0]
    imgB = image[1]
    real_A.data.copy_(imgA)
    fake = netG(real_A)
    fake10 = fake.cpu().data.numpy()
    fake = (fake.cpu().data.numpy() > 0.1 ) * 1
    target = imgB.numpy()
    #target = (target > 0.1) * 1

    l1 = []
    l2 = []
    l3 = []
    
#    for num in range(10):
#        pod, far, mis = csi(fake[0][num],target[0][num])
#        l1.append(pod)
#        l2.append(far)
#        l3.append(mis)
#
#    lpod = lpod + np.array(l1)
#    lfar = lfar + np.array(l2)
#    lmis = lfar + np.array(l3)
    
    for n,pic in enumerate(fake10[0]):
        rb = target[0,n]
        L1 = np.abs(pic - rb)
        combine = np.concatenate((pic,rb),1)
        combine = np.concatenate((combine,L1),1)
        misc.imsave('%s/%d_%d.png' % (opt.outf,i,n),combine)
 
    if(i+1 >= opt.imgNum):
        break

    #if i%100==0:
    #    print('\n',i)
    #    print('lcsi:',lpod/(lpod + lfar + lmis))
    #    print('lpod:',lpod/(lpod + lfar))
    #    print('lfar:',lfar/(lpod + lfar))

#print('lcsi:',lpod/(lpod + lfar + lmis))
#print('lpod:',lpod/(lpod + lfar))
#print('lfar:',lfar/(lpod + lfar))
