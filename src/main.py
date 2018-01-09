from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from scipy import misc
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

DIM = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        encode = nn.Sequential(
            nn.Conv2d(1, DIM, 1),
            nn.ReLU(),
            nn.Conv2d(DIM, 2*DIM, 3, padding=1),
            nn.ReLU(),
#            nn.Conv2d(2*DIM, 4*DIM, 3, padding=1),
#            nn.ReLU(),
#            nn.Conv2d(4*DIM, 8*DIM, 3, padding=1),
#            nn.ReLU(),
#            nn.Conv2d(8*DIM, DIM, 3,padding=1),
#            nn.ReLU(),
            nn.Conv2d(2*DIM, 1, 3,padding=1),
            nn.Sigmoid(),
        )
        self.encode = encode


    def forward(self, x):
        x = self.encode(x)
        return x

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            _data, _target = data.cuda(), target.cuda()
        data = _data[:,:,:14,:]
        target = _data[:,:,14:,:]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            _data, _target = data.cuda(), target.cuda()
        data = _data[:,:,:14,:]
        target = _data[:,:,14:,:]
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        fake = torch.cat((data,output),2)
        real = _data
        a = _data.cpu().numpy() 
        b = fake.data.cpu().numpy()
        re = a[:10].reshape((280,28))
        fa = b[:10].reshape((280,28))
        out = np.concatenate((re,fa),axis=1)
        misc.imsave('guess_img/%d.png' % epoch, out)
        break
        



for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
