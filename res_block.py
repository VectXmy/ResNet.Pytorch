__author__="xu hongtao"
__email__="xxxmy@foxmail.com"

import torch.nn as nn
import torch.nn.functional as F


class basic_2d(nn.Module):
    def __init__(self,in_channel,filters,stride=None,kernel_size=3,stage=0,block=0):
        super(basic_2d,self).__init__()
        if stride is None:
            if block!=0 or stage==0:
                stride=1
            else:
                stride=2
        self.residual=nn.Sequential(
                                nn.Conv2d(in_channel,filters,kernel_size,stride=stride,bias=False,padding=1,padding_mode='zeros'),
                                nn.BatchNorm2d(filters,eps=1e-5),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(filters,filters,kernel_size,bias=False,padding=1,padding_mode='zeros'),
                                nn.BatchNorm2d(filters,eps=1e-5)
                                )
        if block ==0:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channel,filters,(1,1),stride=stride,bias=False),
                nn.BatchNorm2d(filters,eps=1e-5)
            )
        else:
            self.shortcut=None
    

    def forward(self,x):
        residual=self.residual(x)
        if self.shortcut is None:
            shortcut=x
        else:
            shortcut=self.shortcut(x)
        
        y=residual+shortcut
        y=F.relu(y,inplace=True)
        # print(y.size())
        return y

class bottlneck_2d(nn.Module):
    def __init__(self,in_channel,filters,stride=None,kernel_size=3,stage=0,block=0):
        super(bottlneck_2d,self).__init__()
        if stride is None:
            if block!=0 or stage==0:
                stride=1
            else:
                stride=2
        self.residual=nn.Sequential(
                                    nn.Conv2d(in_channel,filters,(1,1),stride=stride,bias=False),
                                    nn.BatchNorm2d(filters,eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(filters,filters,kernel_size,bias=False,padding=1,padding_mode='zeros'),
                                    nn.BatchNorm2d(filters,eps=1e-5),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(filters,filters*4,(1,1),bias=False),
                                    nn.BatchNorm2d(filters*4,eps=1e-5)
                                    )
        if block ==0:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channel,filters*4,(1,1),stride=stride,bias=False),
                nn.BatchNorm2d(filters*4,eps=1e-5)
            )
        else:
            self.shortcut=None
    
    def forward(self,x):
        residual=self.residual(x)
        if self.shortcut is None:
            shortcut=x
        else:
            shortcut=self.shortcut(x)
        
        y=residual+shortcut
        y=F.relu(y,inplace=True)
        # print(y.size())
        return y

