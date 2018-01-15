from __future__ import print_function
from __future__ import division
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
from model.Generator import Generator
#from utils.dataset import Facades
import numpy as np
from scipy import misc
import glob
import gzip
import cPickle

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
parser.add_argument('--which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--outf', default='samples/', help='folder to output images and model checkpoints')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--imgNum', type=int, default=32, help='How many images to generate?')

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

flist = glob.glob('%s/*.pkl.gz' % opt.dataPath)


def CSI(a,b):
    img = np.zeros((256,256,3))
    a = (a>0.03) * 1
    b = (b>0) * 1
    c = a-b
    d = a+b
    R = c==1
    G = d==2
    B = c==-1

    value = np.sum(G)/(np.sum(R) + np.sum(G) + np.sum(B))

    img[:,:,0] = R*255
    img[:,:,1] = G*255
    img[:,:,2] = B*255
   
    return img,value
    
    

for i, image in enumerate(flist):
    f = gzip.open(image)
    m = cPickle.load(f)
    m = m/16.0
     
    imgA = torch.Tensor(m[:,:3])
    
    real_A.data.resize_(imgA.size()).copy_(imgA)
    fake1 = netG(real_A)
    fake2 = netG(fake1)
    fake3 = netG(fake2)
    num = 0
    A = real_A.cpu().data.numpy()
    B = fake1.cpu().data.numpy()
    C = fake2.cpu().data.numpy()
    D = fake3.cpu().data.numpy()

    l = []
    ls = []
    for n in range(3):
        d = np.abs(A[0,n] - m[0,num])
        csi,v = CSI(A[0,n],m[0,num])
        csi_s,v_s = CSI(m[0,num],m[0,2])
        s = np.abs(m[0,num] - m[0,2])

        l.append(v)
        ls.append(v_s)

        r = np.concatenate((A[0,n],m[0,num]),1)
        r = np.concatenate((r,d),1)
        r = np.concatenate((r,s),1)
        misc.imsave('%s/%d_%d.png' % (opt.outf,i,num),r)
        #misc.imsave('%s/%d_%d_rgb.png' % (opt.outf,i,num),csi)
        num +=1
    for n in range(3):
        d = np.abs(B[0,n] - m[0,num])
        csi,v = CSI(B[0,n],m[0,num])
        csi_s,v_s = CSI(m[0,num],m[0,2])
        s = np.abs(m[0,num] - m[0,2])

        l.append(v)
        ls.append(v_s)

        r = np.concatenate((B[0,n],m[0,num]),1)
        r = np.concatenate((r,d),1)
        r = np.concatenate((r,s),1)
        misc.imsave('%s/%d_%d.png' % (opt.outf,i,num),r)
        #misc.imsave('%s/%d_%d_rgb.png' % (opt.outf,i,num),csi)
        num +=1
    for n in range(3):
        d = np.abs(C[0,n] - m[0,num])
        csi,v = CSI(C[0,n],m[0,num])
        csi_s,v_s = CSI(m[0,num],m[0,2])
        s = np.abs(m[0,num] - m[0,2])

        l.append(v)
        ls.append(v_s)

        r = np.concatenate((C[0,n],m[0,num]),1)
        r = np.concatenate((r,d),1)
        r = np.concatenate((r,s),1)
        misc.imsave('%s/%d_%d.png' % (opt.outf,i,num),r)
        #misc.imsave('%s/%d_%d_rgb.png' % (opt.outf,i,num),csi)
        num +=1
    for n in range(3):
        d = np.abs(D[0,n] - m[0,num])
        csi,v = CSI(D[0,n],m[0,num])
        csi_s,v_s = CSI(m[0,num],m[0,2])
        s = np.abs(m[0,num] - m[0,2])

        l.append(v)
        ls.append(v_s)

        r = np.concatenate((D[0,n],m[0,num]),1)
        r = np.concatenate((r,d),1)
        r = np.concatenate((r,s),1)
        misc.imsave('%s/%d_%d.png' % (opt.outf,i,num),r)
        #misc.imsave('%s/%d_%d_rgb.png' % (opt.outf,i,num),csi)
        num +=1

    l = np.around(np.array(l),decimals=2)
    ls = np.around(np.array(ls),decimals=2)
     
    print('img ',i)
    print(l)
    print(ls)
    if(i+1 >= opt.imgNum):
        break
