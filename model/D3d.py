import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,input_nc,output_nc,ndf):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(2,ndf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer2 = nn.Sequential(nn.Conv3d(ndf,ndf*2,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer3 = nn.Sequential(nn.Conv3d(ndf*2,ndf*4,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer4 = nn.Sequential(nn.Conv3d(ndf*4,ndf*8,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer5 = nn.Sequential(nn.Conv3d(ndf*8,1,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
                                 nn.Sigmoid())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
