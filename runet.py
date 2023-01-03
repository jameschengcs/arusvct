import numpy as np
import torch
import torch
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import Module, ModuleList, Sequential, Sigmoid, Hardtanh, Conv2d, Linear, MaxPool2d, ReLU, Dropout, AdaptiveAvgPool2d, Flatten, ConvTranspose2d, BatchNorm2d
import numpy as np

# defining model architecture
class DecBlock(Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = Sequential(
                        Conv2d(in_ch, out_ch, kernel_size = 3, stride=1, padding=1), 
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),
                        Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=1),
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),)
    def forward(self, x):
        return self.seq(x) 

# defining model architecture
class EncBlockH(Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = Sequential(
                        Conv2d(in_ch, out_ch, kernel_size = 3, stride=1, padding=1), 
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),
                        Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=1),
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),)
    def forward(self, x):
        return self.seq(x)     
class EncBlock(Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = Sequential(
                        Conv2d(in_ch, out_ch, kernel_size = 3, stride=2, padding=1), 
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),
                        Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=1),
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),
                        Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=1),
                        ReLU(inplace=True),
                        BatchNorm2d(out_ch),                        )
    def forward(self, x):
        return self.seq(x)    


class Encoder(Module):
    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()
        self.enc_block_h = EncBlockH(chs[0], chs[1])
        self.enc_blocks = ModuleList([EncBlock(chs[i], chs[i+1]) for i in range(1, len(chs)-1)])

    def forward(self, x):
        ftrs = []
        x = self.enc_block_h(x)
        ftrs.append(x)
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
        return ftrs


class Decoder(Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = ModuleList([ConvTranspose2d(chs[i], chs[i+1], kernel_size = 4, stride=2, padding=1) for i in range(len(chs)-1)])
        self.dec_blocks = ModuleList([DecBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            x        = torch.cat([x, encoder_features[i]], dim=1)
            x        = self.dec_blocks[i](x)            
        return x


class RUNet(Module):
    def __init__(self, 
                 enc_chs=(1, 32, 64, 128, 256, 512),  
                 dec_chs=(512, 256, 128, 64, 32)
                 #enc_chs=(1, 64, 128, 256, 512, 1024),  
                 #dec_chs=(1024, 512, 256, 128, 64)
                 ):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = Conv2d(dec_chs[-1], 1, 1)

    def forward(self, x):
        #print('SinoNet::forward::x', x.shape)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out += x
        return out

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
