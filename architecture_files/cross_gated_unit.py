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
        self.dw_conv = nn.Conv2d(in_channels=num_channel_in, 
                                 out_channels=num_channel_out, 
                                 kernel_size=kernel_dim, 
                                 stride=stride_size, 
                                 padding=padding_dim, 
                                 groups=num_channel_in, 
                                 bias=True)

    def forward(self, input_x):
        """
        @param input_x: pytorch data tensor
        @returns: depth-wise convolution of input data
        """
        return self.dw_conv(input_x)

class DownsamplingLayer(nn.Module):
    def __init__(self,
                channels=64):
        super(DownsamplingLayer, self).__init__()
        self.down_layer = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, input_x):
        return self.down_layer(input_x)

class CrossGatedUnit(nn.Module):
    def __init__(self, 
                 is_self_gate=False,
                 number_in_channels=3,
                 number_hidden_channels=128,
                 kernel_dim=(3, 3), 
                 stride_size=1,
                 padding_dim=1,
                 use_dropout=False):
        super(CrossGatedUnit, self).__init__()
        self.is_self_gate = is_self_gate
        self.use_dropout = use_dropout

        if use_dropout:
            self.dropout_layer = nn.Dropout()
        self.conv1_left = nn.Conv2d(in_channels=number_in_channels, 
                                    out_channels=number_hidden_channels, 
                                    kernel_size=1, 
                                    stride=stride_size, 
                                    padding=0)
        self.convdw_left = DepthWiseConvolution(num_channel_in=number_hidden_channels, 
                                                num_channel_out=number_hidden_channels, 
                                                kernel_dim=kernel_dim, 
                                                stride_size=stride_size, 
                                                padding_dim=padding_dim)
        self.conv1_right = nn.Conv2d(in_channels=number_in_channels, 
                                     out_channels=number_hidden_channels, 
                                     kernel_size=1, 
                                     stride=stride_size, 
                                     padding=0)
        self.convdw_right = DepthWiseConvolution(num_channel_in=number_hidden_channels, 
                                                 num_channel_out=number_hidden_channels, 
                                                 kernel_dim=kernel_dim, 
                                                 stride_size=stride_size, 
                                                 padding_dim=padding_dim)
        self.final_conv = nn.Conv2d(in_channels=number_hidden_channels, 
                                    out_channels=number_in_channels, 
                                    kernel_size=1, 
                                    stride=stride_size, 
                                    padding=0)
        self.gelu = nn.GELU()
        if is_self_gate: # self-gate
            self.conv1_cross_gate = None
        else: # cross-gate
            self.conv1_cross_gate = nn.Conv2d(in_channels=int(number_in_channels*2), 
                                              out_channels=number_in_channels, 
                                              kernel_size=1, 
                                              stride=stride_size, 
                                              padding=0)

    def forward(self, input_x, input_y=None):
        """
        @param input_x: Pytorch tensor of dims: [batch size, channel size, height, width] <--- FIXME: double check for accuracy
        @param input_y: Pytorch tensor of dims: [batch size, channel size, height, width] <--- FIXME: double check for accuracy
        @returns: 
        """
        if not self.is_self_gate: # cross-gate performs an extra convolution on the two inputs
            output_xy = torch.cat((input_x, input_y), dim=1)  # concatenate along channel dimension
            output_xy = self.conv1_cross_gate(output_xy)

            # Left
            output_xy = self.conv1_left(output_xy)
            output_xy = self.convdw_left(output_xy)
            output_xy = self.gelu(output_xy)

            # Right
            input_y = self.conv1_right(input_y)
            input_y = self.convdw_right(input_y)
            
            output = torch.mul(output_xy, input_y)  # elementwise multiplication along channel dimension
            if self.use_dropout:
                output = self.dropout_layer(output)
            output = self.final_conv(output)  # final convolution
            if self.use_dropout:
                output = self.dropout_layer(output)
            return output
        
        else:  # self-gate only
            # Left CGU conv1/convDW
            output_left = self.conv1_left(input_x)
            # Right CGU conv1
            output_right = self.conv1_right(input_x)
            
            output_left = self.convdw_left(output_left)
            # FIXME: Add layer scale?
            output_left = self.gelu(output_left)

            
            # Right CGU convDW
            output_right = self.convdw_right(output_right)

            # Elementwise multiplication
            output = torch.mul(output_left, output_right)

            # Final convolution and dropout x2
            if self.use_dropout:
                output = self.dropout_layer(output)
            output = self.final_conv(output)
            if self.use_dropout:
                output = self.dropout_layer(output)
            return output
            
class CrossGateBlock(nn.Module):
    def __init__(self, num_channels_in, 
                 num_channels_hidden, 
                 norm_type="batch_norm", 
                 use_dropout=True, 
                 use_layer_scale=False):
        super(CrossGateBlock, self).__init__()

        self.use_dropout = use_dropout
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            pass  # FIXME for layer scale
        if use_dropout:
            self.dropout_layer = nn.Dropout()
        else:
            self.dropout_layer = None
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(num_features=num_channels_in, affine=False)
        else:  # use group_norm
            self.norm = nn.GroupNorm(num_groups=8, num_channels=num_channels_in, affine=False)

        # Instantiate CGU self and cross units
        self.self_gate_left = CrossGatedUnit(is_self_gate=True,
                                             number_in_channels=num_channels_in,
                                             number_hidden_channels=num_channels_hidden,
                                             kernel_dim=(7, 7), 
                                             stride_size=1,
                                             padding_dim=3,
                                             use_dropout=use_dropout)
        self.self_gate_right = CrossGatedUnit(is_self_gate=True,
                                             number_in_channels=num_channels_in,
                                             number_hidden_channels=num_channels_hidden,
                                             kernel_dim=(7, 7), 
                                             stride_size=1,
                                             padding_dim=3,
                                             use_dropout=use_dropout)
        self.cross_gate_left = CrossGatedUnit(is_self_gate=False,
                                             number_in_channels=num_channels_in,
                                             number_hidden_channels=num_channels_hidden,
                                             kernel_dim=(7, 7), 
                                             stride_size=1,
                                             padding_dim=3,
                                             use_dropout=use_dropout)
        self.cross_gate_right = CrossGatedUnit(is_self_gate=False,
                                             number_in_channels=num_channels_in,
                                             number_hidden_channels=num_channels_hidden,
                                             kernel_dim=(7, 7), 
                                             stride_size=1,
                                             padding_dim=3,
                                             use_dropout=use_dropout)

    def forward(self, x, y):
        x_skip = x.clone()
        y_skip = y.clone()

        # Normalize data prior to CGU forward pass
        x = self.norm(x)
        y = self.norm(y)

        x = self.self_gate_left(x)
        y = self.self_gate_right(y)
        
        x = self.cross_gate_left(x, y)
        y = self.cross_gate_right(x, y)
        
        
        # FIXME: LAYER SCALE?
        if self.use_layer_scale:
            # x = 
            # y = 
            pass
        if self.use_dropout:
            x = self.dropout_layer(x)
            y = self.dropout_layer(y)
        # Skip Connection:
        x = x + x_skip
        y = y + y_skip
        return x, y