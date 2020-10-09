import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class Conv_PSI(nn.Module):
    def __init__(self, num_classes, input_size, input_channel, F1, D, F2, dropout_rate):
        super(mini_EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=D*F1, kernel_size=(32, 1), groups=input_channel, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 16), groups=F1*D, bias=False),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.BN_0 = nn.BatchNorm2d(input_channel)
        self.BN_1 = nn.BatchNorm2d(F1*D)
        self.BN_2 = nn.BatchNorm2d(F2)
        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], num_classes),
        )

    def forward(self, x):
        # x = x[:, 0:5, :, :]
        x = self.BN_0(x)
        y = self.block1(x)
        y = self.BN_1(y)
        y = self.block2(y)
        y = self.BN_2(y)
        y = y.view(y.size()[0], -1)
        out = self.fc1(y)
        return out

    def get_size(self, input_size):
        data = torch.ones(1, input_size[0], input_size[1], input_size[2])
        y = self.BN_0(data)
        y = self.block1(y)
        y = self.BN_1(y)
        y = self.block2(y)
        out = self.BN_2(y)
        out = out.view(out.size()[0], -1)
        return out.size()


