import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1 and in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        tmpx = x
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))

        if self.downsample:
            # There's always this downsample in this model
            tmpx = self.downsample(tmpx)
            tmpx = self.norm(tmpx)

        return self.relu(x + tmpx)


class ResStem(nn.Module):
    def __init__(self, hidden_channels: list[int]):
        """
        This takes hidden_channels, e.g. [64, 96, 128] from DPFlow
        """
        super(ResStem, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, hidden_channels[0], kernel_size=7, stride=2, padding=3) # half the width and hight
        self.norm1 = nn.BatchNorm2d(hidden_channels[0]) # use BatchNorm2d for now (DPFlow use GroupNorm)
        self.res1 = nn.Sequential(ResidualBlock(hidden_channels[0], hidden_channels[0], stride=1),
                                  ResidualBlock(hidden_channels[0], hidden_channels[0], stride=1))

        self.res2 = nn.Sequential(ResidualBlock(hidden_channels[0], hidden_channels[1], stride=2),
                                  ResidualBlock(hidden_channels[1], hidden_channels[1], stride=1))

        self.conv2 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=1) # only change channel size
        
        

    def forward(self, x):
        """
        x: the image
        output: tensor with channels of the hidden_channels[2], e.g. 128
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.conv2(x)

        return x
        