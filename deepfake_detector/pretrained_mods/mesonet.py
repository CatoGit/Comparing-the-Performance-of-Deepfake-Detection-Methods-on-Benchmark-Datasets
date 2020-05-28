"""
    The MesoInception4 deepfake detection architecture
    as introduced in MesoNet: a Compact Facial Video Forgery Detection Network
    (https://arxiv.org/abs/1809.00888) from Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen

    Darius Afchar's TensorFlow implementation [specifies a,b,c,d as [1,4,4,2] and [2,4,4,2] while their paper says [1,4,4,1] and [1,4,4,2]]
    https://github.com/DariusAf/MesoNet/tree/b0d9a5e4bb897f5231558aa182759489a6d22b26

    License (Apache License 2.0) in model_licenses folder

    Ported to PyTorch by Christopher Otto [assumes that a,b,c,d as [1,4,4,2] and [2,4,4,2] 
    are the parameters that were used by the Afchar et al.]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionLayer(nn.Module):
    def __init__(self, num_in_channels, a, b, c, d):
        super(InceptionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_in_channels,
                               out_channels=a, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=num_in_channels,
                               out_channels=b, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=b, out_channels=b,
                               kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=num_in_channels,
                               out_channels=c, kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c,
                               kernel_size=(3, 3), padding=2, dilation=2)
        self.conv6 = nn.Conv2d(in_channels=num_in_channels,
                               out_channels=d, kernel_size=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=d, out_channels=d,
                               kernel_size=(3, 3), padding=3, dilation=3)

    def forward(self, input):
        x1 = F.relu(self.conv1(input))

        x2 = F.relu(self.conv2(input))
        x2 = F.relu(self.conv3(x2))

        x3 = F.relu(self.conv4(input))
        x3 = F.relu(self.conv5(x3))

        x4 = F.relu(self.conv6(input))
        x4 = F.relu(self.conv7(x4))

        x = torch.cat((x1, x2, x3, x4), 1)
        return x


class MesoInception4(nn.Module):

    def __init__(self, num_classes=1):
        # add Python 2 compatibility
        super(MesoInception4, self).__init__()
        # takes a,b,c,d as [1,4,4,2] and [2,4,4,2] similar to Afchar et al.'s implementation
        # but different from the paper specification
        self.inception1 = InceptionLayer(num_in_channels=3, a=1, b=4, c=4, d=2)
        self.bn1 = nn.BatchNorm2d(11)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.inception2 = InceptionLayer(
            num_in_channels=11, a=2, b=4, c=4, d=2)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv1 = nn.Conv2d(
            in_channels=12, out_channels=16, kernel_size=(5, 5), padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=(5, 5), padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d((4, 4))
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=1024, out_features=16)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, input):
        x = F.relu(self.inception1(input))
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(self.inception2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv1(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.conv2(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = x.view(x.shape[0], -1)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.drop2(x)
        x = self.fc2(x)

        return x
