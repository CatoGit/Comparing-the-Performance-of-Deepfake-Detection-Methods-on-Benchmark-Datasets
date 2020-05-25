"""
    The MesoInception4 deepfake detection architecture
    as introduced in MesoNet: a Compact Facial Video Forgery Detection Network
    (https://arxiv.org/abs/1809.00888) from Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen

    Darius Afchar's TensorFlow implementation 
    https://github.com/DariusAf/MesoNet/tree/b0d9a5e4bb897f5231558aa182759489a6d22b26

    License (Apache License 2.0) in model_licenses folder

    Ported to PyTorch by Christopher Otto 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionLayer(nn.Module):
    def __init__(self,a,b,c,d):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=a, kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=(1,1))
        self.conv3 = nn.Conv2d(in_channels=b, out_channels=b, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(in_channels=b, out_channels=c, kernel_size=(1,1))
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(3,3), dilation=2)
        self.conv6 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=(1,1))
        self.conv7 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=(3,3), dilation=3)
    
    def forward(self,input):
        x1 = F.relu(self.conv1(input))

        x2 = F.relu(self.conv2(input))
        x2 = F.relu(self.conv3(x2))

        x3 = F.relu(self.conv4(input))
        x3 = F.relu(self.conv5(x3))

        x4 = F.relu(self.conv6(input))
        x4 = F.relu(self.conv7(x4))
        
        x = torch.cat((x1,x2,x3,x4), -1)
        print(x.shape)
        return x

class MesoInception4(nn.Module):
    
    def __init__(self,num_classes=1):
        # add Python 2 compatibility
        super(MesoInception4, self).__init__()
        self.inception1 = InceptionLayer(a=1,b=4,c=4,d=1)
        self.bn1 = nn.BatchNorm2d(10)
        

    

    def forward(self, input):
        x = self.relu(inception1(input))
        