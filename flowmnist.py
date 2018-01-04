from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from scipy import misc
import pyinn as P

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

dataset = np.load('data/mnist_test_seq.npy')
dataset = dataset/255.0
traindata = dataset[:3]
traindata = traindata.transpose(1,0,2,3)
traindata = torch.Tensor(traindata)
testdata  = dataset[:3,9000:]
testdata  = testdata.transpose(1,0,2,3)
testdata = torch.Tensor(testdata)


train_loader = torch.utils.data.DataLoader(traindata,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testdata,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        block = nn.Sequential(
            nn.Conv2d(2, 20, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 20, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(20, 1, kernel_size=5, padding=2),
            nn.ReLU(True),
        )
        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)
        self.block = block

    def forward(self, x):
        flow = self.block(x)
        x = self.conv1(x)
        x = P.im2col(x,5,1,0)
        flow = P.im2col(flow,5,1,0)
        x = x * flow
        x = P.col2im(x,5,1,0)
        
        return x 

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, lset in enumerate(train_loader):
        data = lset[:,:2]
        target = lset[:,1]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
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
    for lset in test_loader:
        data = lset[:20,:2]
        target = lset[:20,1]
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        a = output.cpu().data.numpy()
        b = target.cpu().data.numpy()
        print(b.shape)
        for x in range(20):
            ba = np.zeros((128,64))
            ba[:64] = a[x][0]
            ba[64:] = b[x]
            misc.imsave('img/%d_%d.png' % (epoch,x),ba)
            print('img/%d_%d.png' % (epoch,x))
        #test_loss += F.mse_loss(output, target, size_average=False).data[0] # sum up batch loss

    #test_loss /= len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    #if epoch % 10==0:
    test(epoch)
