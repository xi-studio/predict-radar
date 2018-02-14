import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        self.conv0 = nn.Conv2d(input_nc, ngf, 1)
        self.conv1 = nn.Conv2d(ngf, ngf, 15, 1, 7)
        self.conv2 = nn.Conv2d(ngf, output_nc, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        c0 = self.relu(self.conv0(input))
        c1 = self.relu(self.conv1(c0))
        output = self.tanh(self.conv2(c1))

        return output
