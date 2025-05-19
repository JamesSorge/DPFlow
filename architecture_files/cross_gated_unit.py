import torch
import torch.nn as nn
import torchvision

class CrossGatedUnit(nn.Module):
    def __init__(self, is_self_gate=False):
        super(CrossGatedUnit, self).__init__()
        self.is_self_gate = is_self_gate
        self.conv1_left = nn.Conv2d(in_channels=, out_channels=, kernel_size=(,), stride=1, padding=0)
        self.convdw_left = nn.Conv2d(in_channels=, out_channels=, kernel_size=(,), stride=1, padding=0)
        self.conv1_right = nn.Conv2d(in_channels=, out_channels=, kernel_size=(,), stride=1, padding=0)
        self.convdw_right = nn.Conv2d(in_channels=, out_channels=, kernel_size=(,), stride=1, padding=0)
        self.final_conv = nn.Conv2d(in_channels=, out_channels=, kernel_size=(,), stride=1, padding=0)
        self.gelu = nn.GELU()
        if is_self_gate: # self-gate
            self.conv1_cross_gate = None
        else: # cross-gate
            self.conv1_cross_gate = nn.Conv2d(in_channels=, out_channels=, kernel_size=(,), stride=1, padding=0)

    def forward(self, input_x):
        """
        @param input_x: Pytorch tensor of dims: [batch size, channel size, height, widith] <--- FIXME: double check for accuracy
        """
        if not self.is_self_gate: # cross-gate performs an extra convolution on the two inputs
            
            