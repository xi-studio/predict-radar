import torch.nn as nn
from torch.nn import DataParallel

class Discriminator(nn.Module):
    def __init__(self,input_nc,output_nc,ndf):
        super(Discriminator,self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc+output_nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=1,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1),
                                 nn.Sigmoid())
        # 30 x 30
        self.layer1 = DataParallel(self.layer1)
        self.layer2 = DataParallel(self.layer2)
        self.layer3 = DataParallel(self.layer3)
        self.layer4 = DataParallel(self.layer4)
        self.layer5 = DataParallel(self.layer5)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
