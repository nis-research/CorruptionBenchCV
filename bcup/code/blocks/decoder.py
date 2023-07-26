import math

import torch
import torch.nn as nn
 


class Decoder(nn.Module):
    def __init__(self,block_de,in_planes):
        super(Decoder,self).__init__()
        self.in_planes = in_planes

        self.layer6 = self._make_uplayer(block_de,256,2,stride=2)
        self.layer7 = self._make_uplayer(block_de,128,2, stride=2)
        self.layer8 = self._make_uplayer(block_de,64,2,stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,3,kernel_size=3,padding=1),
            nn.Sigmoid()
        )
    def _make_uplayer(self, block,planes,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layer6(x)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.conv2(out)
        return out

