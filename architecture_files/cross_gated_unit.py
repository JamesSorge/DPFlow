import torch
import torch.nn as nn
import torchvision

class DepthWiseConvolution(nn.Module):
    def __init__(self, num_channel_in, num_channel_out, kernel_dim, stride_size, padding_dim):
        """
        @param num_channel_in: 
        @param num_channel_out:
        @param kernel_dim: tuple, e.g. (3, 3)
        @param stride_size: integer, e.g. 1
        @param padding_dim: integer, e.g. 1
        """
        super(DepthWiseConvolution, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels=num_channel_in, out_channels=num_channel_out, kernel_size=kernel_dim, stride=stride_size, padding=padding_dim, groups=num_channel_in, bias=True)

    def forward(self, input_x):
        """
        @param input_x: pytorch data tensor
        @returns: depth-wise convolution of input data
        """
        return self.dw_conv(input_x)

class CrossGatedUnit(nn.Module):
    def __init__(self, is_self_gate=False,
                 number_in_channels=3,
                 number_out_channels=128,
                 kernel_dim=(3, 3), 
                 stride_size=1,
                 padding_dim=1):
        super(CrossGatedUnit, self).__init__()
        self.is_self_gate = is_self_gate
        self.conv1_left = nn.Conv2d(in_channels=number_in_channels, out_channels=number_out_channels, kernel_size=kernel_dim, stride=stride_size, padding=padding_dim)
        self.convdw_left = DepthWiseConvolution(num_channel_in=number_out_channels, num_channel_out=number_out_channels, kernel_dim=kernel_dim, stride_size=stride_size, padding_dim=padding_dim)
        self.conv1_right = nn.Conv2d(in_channels=number_out_channels, out_channels=number_out_channels, kernel_size=kernel_dim, stride=stride_size, padding=padding_dim)
        self.convdw_right = DepthWiseConvolution(num_channel_in=number_out_channels, num_channel_out=number_out_channels, kernel_dim=kernel_dim, stride_size=stride_size, padding_dim=padding_dim)
        self.final_conv = nn.Conv2d(in_channels=number_out_channels, out_channels=number_out_channels, kernel_size=kernel_dim, stride=stride_size, padding=padding_dim)
        self.gelu = nn.GELU()
        if is_self_gate: # self-gate
            self.conv1_cross_gate = None
        else: # cross-gate
            self.conv1_cross_gate = nn.Conv2d(in_channels=number_out_channels, out_channels=number_out_channels, kernel_size=kernel_dim, stride=stride_size, padding=padding_dim)

    def forward(self, input_x):
        """
        @param input_x: Pytorch tensor of dims: [batch size, channel size, height, widith] <--- FIXME: double check for accuracy
        @returns: 
        """
        if not self.is_self_gate: # cross-gate performs an extra convolution on the two inputs
            input_x = torch.cat((), dim=)
            input_x = self.conv1_cross_gate(input_x)
        else:  # self-gate only
            # FIXME: Is this the correct way to copy this tensor? Am I detaching from the computation graph too early? See https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
            input2 = input_x.clone().detach()

            # Left CGU conv1/convDW
            output_left = self.conv1_left(input_x)
            output_left = self.convdw_left(output_left)
            output_left = self.gelu(output_left)

            # Right CGU conv1/convDW
            output_right = self.conv1_right(input2)
            output_right = self.convdw_right(output_right)

            # Elementwise multiplication
            output = torch.mul(output_left, output_right)

            # Final convolution
            output = self.final_conv(output)

        return output
            
            