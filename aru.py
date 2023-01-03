# This python code is based on the implementation of LeeJunHyun
# See https://github.com/LeeJunHyun/Image_Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

negative_slope = 0.2
lrelu = True
def actf():
    return nn.LeakyReLU(negative_slope = negative_slope, inplace=True) if lrelu else nn.ReLU(inplace=True)

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

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            actf(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            actf(),
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class conv_down_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_down_block,self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=2,padding=1,bias=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=2,stride=2,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            actf(),
        )
    def forward(self,x):
        x = self.conv(x)
        return x        

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
            actf(),
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = actf()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class ARUNet(nn.Module):
    def __init__(self, ch0 = 64, atten_chx = 1):
        super().__init__()
        
        ch_in, ch_out = 1, ch0
        self.Conv1 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down1 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down2 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down3 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv4 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down4 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv5 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up5 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att5 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv5 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up4 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att4 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv4 = conv_block(ch_in = ch_in, ch_out = ch_out)
        
        ch_in, ch_out = ch_out, ch_out // 2
        self.Up3 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att3 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        
        ch_in, ch_out = ch_out, ch_out // 2
        self.Up2 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att2 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)        

        self.Conv_1x1 = nn.Conv2d(ch_out, 1, kernel_size = 1, stride = 1, padding = 0)   
    

    def forward(self,x):
        # encoding
        x1 = self.Conv1(x)

        x2 = self.Down1(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Down2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Down3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Down4(x4)
        x5 = self.Conv5(x5)

        # decoding
        d5 = self.Up5(x5)
        x4 = self.Att5(g = d5, x = x4)
        d5 = torch.cat((x4, d5), dim = 1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g = d4, x = x3)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g = d3, x = x2)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g = d2, x = x1)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        d1 += x
        return d1
# @ARUNet  

class DUNet(nn.Module):
    def __init__(self, ch0 = 64):
        super().__init__()
        
        ch_in, ch_out = 1, ch0
        self.Conv1 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down1 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down2 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down3 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv4 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down4 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv5 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up5 = up_conv(ch_in = ch_in, ch_out = ch_out)        
        self.Up_conv5 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up4 = up_conv(ch_in = ch_in, ch_out = ch_out)        
        self.Up_conv4 = conv_block(ch_in = ch_in, ch_out = ch_out)
        
        ch_in, ch_out = ch_out, ch_out // 2
        self.Up3 = up_conv(ch_in = ch_in, ch_out = ch_out)        
        self.Up_conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        
        ch_in, ch_out = ch_out, ch_out // 2
        self.Up2 = up_conv(ch_in = ch_in, ch_out = ch_out)        
        self.Up_conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)        

        self.Conv_1x1 = nn.Conv2d(ch_out, 1, kernel_size = 1, stride = 1, padding = 0) 
    

    def forward(self,x):
        # encoding
        x1 = self.Conv1(x)

        x2 = self.Down1(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Down2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Down3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Down4(x4)
        x5 = self.Conv5(x5)

        # decoding
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim = 1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)        
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)        
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)        
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        d1 += x
        return d1
# @DUNet        

class ARUNet3(nn.Module):
    def __init__(self, ch0 = 64, atten_chx = 1):
        super().__init__()
        
        ch_in, ch_out = 1, ch0
        self.Conv1 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down1 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down2 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down3 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv4 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up4 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att4 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv4 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up3 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att3 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        
        ch_in, ch_out = ch_out, ch_out // 2
        self.Up2 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att2 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)        

        self.Conv_1x1 = nn.Conv2d(ch_out, 1, kernel_size = 1, stride = 1, padding = 0)   
    

    def forward(self,x):
        # encoding
        x1 = self.Conv1(x)

        x2 = self.Down1(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Down2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Down3(x3)
        x4 = self.Conv4(x4)

        # decoding
        d4 = self.Up4(x4)
        x3 = self.Att4(g = d4, x = x3)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g = d3, x = x2)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g = d2, x = x1)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        d1 += x
        return d1
#@ ARUNet3

class ARUNet2(nn.Module):
    def __init__(self, ch0 = 64, atten_chx = 1):
        super().__init__()
        
        ch_in, ch_out = 1, ch0
        self.Conv1 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down1 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)
        self.Down2 = conv_down_block(ch_in = ch_out, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out + ch_out
        self.Conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)

        ch_in, ch_out = ch_out, ch_out // 2
        self.Up3 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att3 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv3 = conv_block(ch_in = ch_in, ch_out = ch_out)
        
        ch_in, ch_out = ch_out, ch_out // 2
        self.Up2 = up_conv(ch_in = ch_in, ch_out = ch_out)
        self.Att2 = Attention_block(F_g = ch_out, F_l = ch_out, F_int = ch_out // atten_chx)
        self.Up_conv2 = conv_block(ch_in = ch_in, ch_out = ch_out)        

        self.Conv_1x1 = nn.Conv2d(ch_out, 1, kernel_size = 1, stride = 1, padding = 0)   
    

    def forward(self,x):
        # encoding
        x1 = self.Conv1(x)

        x2 = self.Down1(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Down2(x2)
        x3 = self.Conv3(x3)

        # decoding
        d3 = self.Up3(x3)
        x2 = self.Att3(g = d3, x = x2)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g = d2, x = x1)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        d1 += x
        return d1
#@ ARUNet2

