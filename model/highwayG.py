import torch.nn as nn
import torch
from torch.nn import DataParallel

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        self.conv0 = nn.Conv2d(input_nc, input_nc * 3, 1)
        self.conv1 = nn.Conv2d(input_nc * 3, ngf, 3, 1, 1)
        self.conv2 = nn.Conv2d(ngf, ngf, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf, ngf, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf, ngf, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf, ngf, 4, 2, 1)

        self.dconv1 = nn.Conv2d(ngf, output_nc, 3, 1, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1)
        self.dconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1),
        )
        self.dconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1),
        )
        self.dconv5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1),
        )

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        c0 = self.conv0(input)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv4(c2)
        c4 = self.conv5(c3)
        c5 = self.conv5(c4)

        e1 = self.relu(c1)
        e2 = self.relu(c2)
        e3 = self.relu(c3)
        e4 = self.relu(c4)
        e5 = self.relu(c5)

        d1 = self.dconv1(e1)
        d2 = self.dconv2(e2)
        d3 = self.dconv3(e3)
        d4 = self.dconv4(e4)
        d5 = self.dconv5(e5)

        output = self.tanh(d1 + d2 + d3 + d4 + d5)
        return output
