import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2)
        self.conv3 = nn.Conv1d(out_channels, out_channels,kernel_size=kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        # print('con1: ', out1.shape)
        out2 = self.conv2(out1)
        # print('con2: ', out2.shape)
        out3 = self.relu(out2)
        # print('relu: ', out3.shape)
        out4 = self.conv3(out3)
        # print('con3: ', out4.shape)
        out5 = self.bn(out4)
        # print('bn: ', out5.shape)
        out6 = self.relu(out5 + out1)
        # print('relu: ', out6.shape)
        return out6


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ResNet 1: 3blocks, kernel_size(25*25), dilation=1
        self.ResNet1_block1 = ResidualBlock(21, 64, 25, 1)
        self.ResNet1_block2 = ResidualBlock(64, 64, 25, 1)
        self.ResNet1_block3 = ResidualBlock(64, 64, 25, 1)

        # ResNet 2: 5blocks, kernel_size(5*5), dilations: 1, 2, 4, 8, 16
        self.ResNet2_block1 = ResidualBlock(64, 140, 5, 1)
        self.ResNet2_block2 = ResidualBlock(140, 140, 5, 2)
        self.ResNet2_block3 = ResidualBlock(140, 140, 5, 4)
        self.ResNet2_block4 = ResidualBlock(140, 140, 5, 8)
        self.ResNet2_block5 = ResidualBlock(140, 140, 5, 16)

        self.dropout = nn.Dropout(0.5)
        self.conv70 = nn.Conv1d(140, 70, 1)
        self.conv3 = nn.Conv1d(70, 3, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.float()
        # ResNet 1, 3blocks
        out1 = self.ResNet1_block1(x)
        out2 = self.ResNet1_block2(out1)
        out3 = self.ResNet1_block3(out2)
        # ResNet 2, 5blocks
        out4 = self.ResNet2_block1(out3)
        out5 = self.ResNet2_block2(out4)
        out6 = self.ResNet2_block3(out5)
        out7 = self.ResNet2_block4(out6)
        out8 = self.ResNet2_block5(out7)
        # dropout
        out9 = self.dropout(out8)
        out10 = self.conv70(out9)
        out11 = F.elu(out10)
        out12 = self.conv3(out11)
        out13 = out12.permute(0, 2, 1)
        return out13