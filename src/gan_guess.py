import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 392 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())

# ==================Definition Start======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )

        main = nn.Sequential(
            nn.Conv2d(1, 2*DIM, 5, padding=2),
            nn.BatchNorm2d(2*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 2*DIM, 5, padding=2),
            nn.BatchNorm2d(2*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 1, 5, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.view(-1, 1, 14, 28)
        output = self.main(input)
        #print output.size()
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(2*DIM),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(4*DIM),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input, label):
        input = torch.cat((input,label),1)
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        'tmp/mnist/samples_{}.png'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

def calc_gradient_penalty(netD, real, fake, tag):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    real_data = real 
    fake_data = fake

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
        tag = tag.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    tag = autograd.Variable(tag, requires_grad=True)

    disc_interpolates = netD(interpolates,tag)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
print netG
print netD

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

data = inf_train_gen()

for iteration in xrange(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in xrange(CRITIC_ITERS):
        _data = data.next()
        input_tag = _data[:,:392]
        output_tag = _data[:,392:]
        input_tag = torch.Tensor(input_tag)
        output_tag = torch.Tensor(output_tag)
       
        if use_cuda:
            input_tag = input_tag.cuda(gpu)
            output_tag = output_tag.cuda(gpu)
        
        input_real = autograd.Variable(input_tag)
        output_real = autograd.Variable(output_tag)

        netD.zero_grad()

        # train with real
        D_real = netD(input_real,output_real)
        D_real = D_real.mean()
        # print D_real
        D_real.backward(mone)

        # train with fake
        input_fake = autograd.Variable(input_tag,volatile=True)
        fake = autograd.Variable(netG(input_fake).data)

        D_fake = netD(fake,output_real)
        D_fake = D_fake.mean()
        D_fake.backward(one)
       
        gradient_penalty = calc_gradient_penalty(netD, input_tag, fake.data,output_tag)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty

        Wasserstein_D = D_real - D_fake
        print Wasserstein_D.cpu().data.numpy()
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    _data = data.next()
    input_tag = _data[:,:392]
    output_tag = _data[:,392:]
    input_tag = torch.Tensor(input_tag)
    output_tag = torch.Tensor(output_tag)
    if use_cuda:
        input_tag = input_tag.cuda(gpu)
        output_tag = output_tag.cuda(gpu)
    input_tag = autograd.Variable(input_tag)
    output_tag = autograd.Variable(output_tag)

    fake = netG(input_tag)
    G = netD(fake,output_tag)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    # Write logs and save samples
    lib.plot.plot('tmp/mnist/time', time.time() - start_time)
    lib.plot.plot('tmp/mnist/train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot('tmp/mnist/train gen cost', G_cost.cpu().data.numpy())
    lib.plot.plot('tmp/mnist/wasserstein distance', Wasserstein_D.cpu().data.numpy())

#    # Calculate dev loss and generate samples every 100 iters
#    if iteration % 100 == 99:
#        dev_disc_costs = []
#        for images,_ in dev_gen():
#            imgs = torch.Tensor(images)
#            if use_cuda:
#                imgs = imgs.cuda(gpu)
#            imgs_v = autograd.Variable(imgs, volatile=True)
#
#            D = netD(imgs_v)
#            _dev_disc_cost = -D.mean().cpu().data.numpy()
#            dev_disc_costs.append(_dev_disc_cost)
#        lib.plot.plot('tmp/mnist/dev disc cost', np.mean(dev_disc_costs))
#
#        generate_image(iteration, netG)
#
#    # Write logs every 100 iters
#    if (iteration < 5) or (iteration % 100 == 99):
#        lib.plot.flush()
#
#    lib.plot.tick()
