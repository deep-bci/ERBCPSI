import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

###################################################mini_EEGNet##########################################################
class mini_EEGNet(nn.Module):
    def __init__(self, num_classes, input_size, input_channel, F1, D, F2, dropout_rate):
        super(mini_EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=D*F1, kernel_size=(32, 1), groups=input_channel, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 8), groups=F1*D, bias=False),
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

#####################################################LSTM_Net###########################################################
class LSTM_Net(nn.Module):
    def __init__(self, input_feature=32, hidden_feature=64, num_class=2, num_layers=2):
        super(LSTM_Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_feature,
            hidden_size = hidden_feature,
            num_layers = num_layers,
            # batch_first = True,
        )
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        x = x.squeeze()

        x = x.permute(2, 0, 1)
        out, _ = self.lstm(x)
        out = self.classifier(out[-1, :, :])
        return out

#############################################mini_EEGNet+LSTM#######################################################
class EEGNet_LSTM(nn.Module):
    def __init__(self, num_classes, EEGNet_inputsize, LSTM_inputsize, input_channel, F1, D, F2, dropout_rate,
                 input_feature=32, hidden_feature=64, num_layers=2):
        super(EEGNet_LSTM, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=D * F1, kernel_size=(32, 1), groups=input_channel,
                      bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(in_channels=F1 * D, out_channels=F1 * D, kernel_size=(1, 8), groups=F1 * D, bias=False),
            nn.Conv2d(in_channels=F1 * D, out_channels=F2, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.BN_0 = nn.BatchNorm2d(input_channel)
        self.BN_1 = nn.BatchNorm2d(F1 * D)
        self.BN_2 = nn.BatchNorm2d(F2)

        self.lstm = nn.LSTM(
            input_size=input_feature,
            hidden_size=hidden_feature,
            num_layers=num_layers,
            # batch_first = True,
        )

        size = self.get_size(EEGNet_inputsize, LSTM_inputsize)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], num_classes),
        )


        self.classifier = nn.Linear(size[1], num_classes)

    def forward(self, x):
        EEGNet_x = x[:, 0:5, :, :]
        LSTM_x = x[:, -1, :, :]
        EEGNet_y = self.BN_0(EEGNet_x)
        EEGNet_y = self.block1(EEGNet_y)
        EEGNet_y = self.BN_1(EEGNet_y)
        EEGNet_y = self.block2(EEGNet_y)
        EEGNet_out = self.BN_2(EEGNet_y)
        EEGNet_out = EEGNet_out.view(EEGNet_out.size()[0], -1)

        LSTM_x = LSTM_x.squeeze()
        LSTM_x = LSTM_x.permute(2, 0, 1)
        LSTM_out, _ = self.lstm(LSTM_x)
        LSTM_out = LSTM_out.view(LSTM_out.size()[1], -1)

        out = torch.cat((EEGNet_out, LSTM_out), dim = 1)

        out = self.classifier(out)
        return out

    def get_size(self, EEGNet_inputsize, LSTM_inputsize):
        EEGNet_data = torch.ones((2, EEGNet_inputsize[1], EEGNet_inputsize[2], EEGNet_inputsize[3]))
        LSTM_data = torch.ones((2, 1, LSTM_inputsize[1], LSTM_inputsize[2]))
        EEGNet_y = self.BN_0(EEGNet_data)
        EEGNet_y = self.block1(EEGNet_y)
        EEGNet_y = self.BN_1(EEGNet_y)
        EEGNet_y = self.block2(EEGNet_y)
        EEGNet_out = self.BN_2(EEGNet_y)
        EEGNet_out = EEGNet_out.view(EEGNet_out.size()[0], -1)

        LSTM_data = LSTM_data.squeeze()
        LSTM_data = LSTM_data.permute(2, 0, 1)
        print(LSTM_data.type())
        LSTM_out, _ = self.lstm(LSTM_data)
        LSTM_out = LSTM_out.view(LSTM_out.size()[1], -1)
        # LSTM_out = np.reshape(LSTM_out, (LSTM_out.shape[1], -1))
        out = torch.cat((EEGNet_out, LSTM_out), dim=1)
        out = out.view(out.size()[0], -1)
        return out.size()
################################################Raw_EEGNet##################################
class Raw_EEGNet(nn.Module):
    def __init__(self, num_classes, input_size, n_channel, sample_rate, F1, D, F2, dropout_rate):
        super(Raw_EEGNet, self).__init__()
        self.dropout_rate = dropout_rate
        # Block 1
        # nn.Conv2d(in_channels, out_channels, kernel_size)
        # padding to implement mode='same'
        self.padding_1 = nn.ZeroPad2d((sample_rate//4-1, sample_rate//4, 0, 0))
        self.conv_1 = nn.Conv2d(1, F1, (1, sample_rate))
        self.batchnorm_1 = nn.BatchNorm2d(F1, False)
        self.depthwise_1 = nn.Conv2d(F1, D*F1, (n_channel, 1), groups=F1)
        self.batchnorm_2 = nn.BatchNorm2d(D*F1, False)

        # to reduce the sampling rate of the signal from 128 to 32
        self.avgpool_1 = nn.AvgPool2d(1, 8)
        # Dropout

        # Depthwise separable 2D convolution: Separable convolutions consist in first performing a depthwise spatial convolution
        # (which acts on each input channel separately) followed by a pointwise convolution
        # which mixes together the resulting output channels
        self.padding_2 = nn.ZeroPad2d((sample_rate//16-1, sample_rate//16, 0, 0))
        self.separate_1 = nn.Conv2d(D*F1, D*F1, (1, sample_rate//8), groups=F1*D, bias=False)
        self.separate_2 = nn.Conv2d(D*F1, F2, 1, bias=False)

        self.batchnorm_3 = nn.BatchNorm2d(D*F1, False)
        self.avgpool_2 = nn.AvgPool2d(1, 8)

        size = self.getsize(input_size)
        self.fc = nn.Linear(size[1], num_classes)

    def forward(self, x):
        x = self.padding_1(x)
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.depthwise_1(x)
        x = self.batchnorm_2(x)
        x = F.elu(x)
        x = self.avgpool_1(x)
        x = F.dropout(x, self, self.dropout_rate)
        x = self.fc(x.view(x.shape[0], -1))
        return x


    def getsize(self, input_size):
        data = torch.ones(1, input_size[1], input_size[2], input_size[3])
        x = self.padding_1(data)
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.depthwise_1(x)
        x = self.batchnorm_2(x)
        x = F.elu(x)
        x = self.avgpool_1(x)
        x = F.dropout(x, self, self.dropout_rate)
        out = x.view(x.shape[0], -1)
        return out.size()

