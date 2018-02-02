import torch.nn as nn

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

class Discriminator(nn.Module):
    def __init__(self,input_nc,output_nc,ndf):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc+output_nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=1,padding=1),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1),
                                 nn.Sigmoid())

        self.block1 = ResBlock(ndf*1)
        self.block2 = ResBlock(ndf*2)
        self.block3 = ResBlock(ndf*4)
        self.block4 = ResBlock(ndf*8)

    def forward(self,x):
        out = self.block1(self.layer1(x))
        out = self.block2(self.layer2(out))
        out = self.block3(self.layer3(out))
        out = self.block4(self.layer4(out))
        out = self.layer5(out)
        return out
