#Credit: code copied from https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/models.py
import torch.nn as nn
import torch

class ResBlock(nn.Module):
  
    def __init__(self, DIM):
        super(ResBlock, self).__init__()
        self.DIM = DIM

        self.res_block = nn.Sequential(
            nn.Conv2d(self.DIM, self.DIM, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.DIM, self.DIM, 3, 1, 1),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)

        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)


        self.eblock1 = ResBlock(ngf * 1)
        self.eblock2 = ResBlock(ngf * 2)
        self.eblock3 = ResBlock(ngf * 4)
        self.eblock4 = ResBlock(ngf * 8)
        self.eblock5 = ResBlock(ngf * 8)
        self.eblock6 = ResBlock(ngf * 8)
        self.eblock7 = ResBlock(ngf * 8)
        self.eblock8 = ResBlock(ngf * 8)

        self.dblock1 = ResBlock(ngf * 8)
        self.dblock2 = ResBlock(ngf * 8)
        self.dblock3 = ResBlock(ngf * 8)
        self.dblock4 = ResBlock(ngf * 8)
        self.dblock5 = ResBlock(ngf * 4)
        self.dblock6 = ResBlock(ngf * 2)
        self.dblock7 = ResBlock(ngf * 1)
        self.dblock8 = ResBlock(output_nc)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        #Encoder
        e1 = self.eblock1(self.conv1(input))
        e2 = self.eblock2(self.conv2(self.leaky_relu(e1)))
        e3 = self.eblock3(self.conv3(self.leaky_relu(e2)))
        e4 = self.eblock4(self.conv4(self.leaky_relu(e3)))
        e5 = self.eblock5(self.conv5(self.leaky_relu(e4)))
        e6 = self.eblock6(self.conv6(self.leaky_relu(e5)))
        e7 = self.eblock6(self.conv7(self.leaky_relu(e6)))
        e8 = self.eblock8(self.conv8(self.leaky_relu(e7)))

        # Decoder
        d1_ = self.dblock1(self.dropout(self.dconv1(self.relu(e8))))
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dblock2(self.dropout(self.dconv2(self.relu(d1))))
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dblock3(self.dropout(self.dconv3(self.relu(d2))))
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.dblock4(self.dconv4(self.relu(d3)))
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.dblock5(self.dconv5(self.relu(d4)))
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.dblock6(self.dconv6(self.relu(d5)))
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.dblock7(self.dconv7(self.relu(d6)))
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dblock8(self.dconv8(self.relu(d7)))

        output = self.tanh(d8)
        return output
