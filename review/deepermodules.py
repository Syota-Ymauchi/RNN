import torch
import torch.nn as nn
import torch.nn.functional as F

def usable_module():
    print('ResidualBlock, PreActivationResidualBlock,BottleneckStracture')


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, activation='relu'):
        super().__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError('not support your activation. Choose from ["relu", "swish"]')
        self.main_conv = nn.Sequential(        
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            self.activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        
        self.shortcut = nn.Sequential()
        
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
           
    def forward(self, x):
        out = self.main_conv(x) 
        out += self.shortcut(x) 
        out = self.activation(out)
        return out
    


class PreActivationResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, activation='relu'):
        super().__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError('not support your activation. Choose from ["relu", "swish"]')
        self.main_conv = nn.Sequential(        
            nn.BatchNorm2d(in_ch),
            self.activation,
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            self.activation,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        self.shortcut = nn.Sequential()
        
        if in_ch != out_ch or stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch)
            )
           
    def forward(self, x):
        out = self.main_conv(x) 
        out += self.shortcut(x) 
        out = self.activation(out)
        return out

class BottleneckStracture(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation='relu'):
        super().__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError('not support your activation. Choose from ["relu", "swish"]')
    
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            self.activation,
            nn.Conv2d(out_ch, out_ch*4, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch*4),          
        )
        self.shortcut = nn.Sequential()
        
        if in_ch != out_ch*4 or stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch*4, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch*4)
            )
    def forward(self, x):
        out = self.main_conv(x) 
        return out