import torch
from torch import nn
import torch.nn.functional as F
import sys

path_list = sys.path
found_directories = False
for path_variable in path_list:
    if "DPFlow" in path_variable:
        found_directories = True
    else:
        pass

if not found_directories:
    print(f"Adding */DPFlow/architecture_files to sys.path")
    sys.path.append("/home/jsorge/private/final_project/DPFlow")
    # sys.path.append("/home/yix050/private/ECE285_Visual_25Spring/final_project/DPFlow")
    print(f"sys.path = ")
    for idx in range(len(sys.path)):
        print(f"\t{sys.path[idx]}") 
from architecture_files.cross_gated_unit import CrossGateBlock, DownsamplingLayer
from architecture_files.res_stem import ResStem
from architecture_files.conv_gru_cell import ConvGRUCell


class BidirEncoder(nn.Module):
    """
    The hidden_channels is always [64, 96, 128] in the DPFlow paper, and pyramid_level = 3
    """
    def __init__(self,
                 hidden_channels=[64, 96, 128]):
        super().__init__()

        ### Conv Stem: takes the raw image as input and outputs X0 and H0 ###
        self.conv_stem = ResStem([hidden_channels[0], hidden_channels[1], 2 * hidden_channels[2]])
        self.lower_stem = ResStem([hidden_channels[0], hidden_channels[1], hidden_channels[2]])

        ### Forward GRU ###
        self.forward_gru = ConvGRUCell(hidden_channels[-1], hidden_channels[-1])
        # for passing Hf to the lower scale level. H_out = H_in/2
        self.down_gru = nn.Conv2d(hidden_channels[-1], hidden_channels[-1], kernel_size=3, stride=2, padding=1, bias=True) 

        ### Backward GRU ###
        self.backward_gru = ConvGRUCell(hidden_channels[-1], hidden_channels[-1])
        # for passing Hb and Xb to the higher scale level. H_out = 2*H_in
        self.up_gru = nn.ConvTranspose2d(hidden_channels[-1], hidden_channels[-1], kernel_size=4, stride=2, padding=1, bias=True) 

        ### Forward CGU ###
        self.downsampling = DownsamplingLayer(channels=hidden_channels[-1])
        self.forward_cgu = CrossGateBlock(num_channels_in=hidden_channels[-1], 
                                          num_channels_hidden=hidden_channels[-1], 
                                          norm_type="batch_norm", 
                                          use_dropout=True, 
                                          use_layer_scale=False)
        
        ### Backward CGU ###
        self.backward_cgu = CrossGateBlock(num_channels_in=hidden_channels[-1], 
                                          num_channels_hidden=hidden_channels[-1], 
                                          norm_type="batch_norm", 
                                          use_dropout=True, 
                                          use_layer_scale=False)

        ### Output layer ###
        self.num_out_stages = 1 # DPFlow always sets this to 1
        self.out_1x1_abs_chs = 384 # DPFlow always sets this to 384
        self.out_1x1_factor = None

        if self.num_out_stages > 0:
            # This out_merge_conv has a ReLU layer before it
            self.out_merge_conv = nn.Conv2d(3 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
            self.out_cgu = CrossGateBlock(num_channels_in=hidden_channels[-1], 
                                          num_channels_hidden=hidden_channels[-1], 
                                          norm_type="batch_norm", 
                                          use_dropout=True, 
                                          use_layer_scale=False)
        if self.out_1x1_abs_chs > 0:
            self.out_1x1 = nn.Conv2d(hidden_channels[-1], self.out_1x1_abs_chs, kernel_size=1)



    def forward(self, x: torch.tensor, y: torch.tensor, pyr_levels: int):
        """
        Takes two frames of image x and y as input. x and y will go through the same process separately.
        @param x: a raw image
        @param y: a raw image

        @return: Two feature pyramids x_pyramid[::-1], y_pyramid[::-1] as the embeddings to pass to the decode
        """

        # input_x = x # for concatenation in the final feature pyramid
        # input_y = y # for concatenation in the final feature pyramid
        
        x0, hx0 = self._get_init_stat(x)
        y0, hy0 = self._get_init_stat(y)
        
        x_pyramid, y_pyramid = self._encode(x0, hx0, y0, hy0, pyr_levels, x, y)
        
        return x_pyramid[::-1], y_pyramid[::-1]

    
    def _encode(self, x0, hx0, y0, hy0, pyr_levels, input_x, input_y):
        #TODO: Implement the dual-pyramid encoder block
        """
        Go through the forward process (high scale to low scale) and then the backward process (low scale to high scale)
        Returns the feature pyramids x_pyramid and y_pyramid
        """
        
        x_pyramid = [None] * 3 # store concatenation of xf, xb, xi for each scale
        y_pyramid = [None] * 3 # store concatenation of yf, yb, yi for each scale
        
        # input_x = x0 # for concatenation in the final feature pyramid
        # input_y = y0 # for concatenation in the final feature pyramid
        
        ####### Forward Start #######
        x_forwards = []
        y_forwards = []

        x_f, hx_f = x0, hx0
        y_f, hy_f = y0, hy0
        
        for i in range(pyr_levels):
            hx_f = self.forward_gru(x_f, hx_f)
            hy_f = self.forward_gru(y_f, hy_f)

            x_f, y_f = self.forward_cgu(hx_f, hy_f) # this is used as xf for the next scale, and in the final concatenation
            x_f = self.downsampling(x_f)
            y_f = self.downsampling(y_f)
            x_f = x_f.contiguous() # make the data contiguous to speed up the computation?
            y_f = y_f.contiguous()

            # downsample the forward h
            if (i < pyr_levels - 1): # don't do the down_gru for the lowest level
                hx_f = torch.tanh(self.down_gru(hx_f))
                hy_f = torch.tanh(self.down_gru(hy_f))

            x_forwards.append(x_f)
            y_forwards.append(y_f)
        ####### Forward End #######
        

        ####### Backward Start #######
        # Initialize the hx and hy when doing backward from the lowest scale layer
        hx_b = torch.zeros_like(x_forwards[-1])
        hy_b = torch.zeros_like(y_forwards[-1])

        for i in range(len(x_forwards) - 1, -1, -1):
            x_f = x_forwards[i]
            y_f = y_forwards[i]

            hx_b = self.backward_gru(x_f, hx_b)
            hy_b = self.backward_gru(y_f, hy_b)

            x_b, y_b = self.backward_cgu(hx_b, hy_b)

            # Downsample the input for concatenating with different scales, (1/2, 1/4, 1/8) for each scale.
            input_x_lower = F.interpolate(input_x, scale_factor=(1 / 2**(i+1)), mode='bilinear', align_corners=True)
            input_y_lower = F.interpolate(input_y, scale_factor=(1 / 2**(i+1)), mode='bilinear', align_corners=True)
            
            input_x_lower = self.lower_stem(input_x_lower)
            input_y_lower = self.lower_stem(input_y_lower)

            # These pyramids are the output of this Encoder
            # Sizes of tensors must match except in dimension 1 (height and width must match, will concate along the channel).
            x_pyramid[i] = torch.cat([x_f, x_b, input_x_lower], 1)
            y_pyramid[i] = torch.cat([y_f, y_b, input_y_lower], 1) 
            
            # upsample the backward h
            if i > 0:
                hx_b = torch.tanh(self.up_gru(hx_b))
                hy_b = torch.tanh(self.up_gru(hy_b))
        ####### Backward End #######

        ####### Output Layers ######
        for i, (x, y) in enumerate(zip(x_pyramid, y_pyramid)):
            if self.num_out_stages > 0:
                x = self.out_merge_conv(F.relu(x))
                y = self.out_merge_conv(F.relu(y))
                x, y = self.out_cgu(x, y)
            if self.out_1x1_abs_chs > 0:
                if self.out_1x1_factor is None:
                    x = self.out_1x1(x)
                    y = self.out_1x1(y)
                else:
                    x = self.out_1x1(x, int(self.out_1x1_factor * x.shape[1]))
                    y = self.out_1x1(y, int(self.out_1x1_factor * y.shape[1]))
            x_pyramid[i] = x
            y_pyramid[i] = y
        
        
        return x_pyramid, y_pyramid


    def _get_init_stat(self, x):
        """
        Pass the input image x to the conv_stem, and return x0, h0
        """
        x = self.conv_stem(x)
        x, hx = torch.split(x, [x.shape[1] // 2, x.shape[1] // 2], 1)
        hx = torch.tanh(hx)
        return x, hx
        