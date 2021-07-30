
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from torch.nn import Softmax


class channelRelation(torch.nn.Module):
    
   

    def __init__(self):
        

        super(channelRelation, self).__init__()
        self.softmax_1 = Softmax(dim=1)
        self.softmax_2 = Softmax(dim=1)
        #self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_re = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,257), stride=1)
        
        

    def forward(self, x):
       
   
        b,c,h,w = x.size()
        x1 = x.view(b,c,-1)     
        x2 = x1.permute(0,2,1)
        co_mat = torch.bmm(-x1, x2)
        co_mat = self.softmax_1(co_mat)
        gap = self.gap(x).view(b,c,-1)
        gap = self.softmax_2(gap)
        rel = torch.cat((co_mat,gap), dim = 2)
        rel = rel.view(b,1,c,c+1)
        final = self.conv_re(rel)       
        final = final.view(b,c,1,1)
        
        return x * final.expand_as(x)




