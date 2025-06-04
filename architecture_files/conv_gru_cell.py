import torch
from torch import nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        """
        A GRU-RNC cell.
        
        @type  input_channel: integer
        @param input_channel: number of input channels
        @type  hidden_channel: integer
        @param hidden_channel: number of the hidden channels 
        """
        super(ConvGRUCell, self).__init__()
        self.conv_Wz = nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1)
        self.conv_Uz = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.conv_Wr = nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1)
        self.conv_Ur = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.conv_W = nn.Conv2d(input_channel, hidden_channel, kernel_size=3, padding=1)
        self.conv_U = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.tensor, h_prev: torch.tensor):
        """
        @rtype:   torch.tensor
        @return:  hidden representation of this ConvGRU cell with shape (batch, channel, height, width)
        """
        z = self.sigmoid(self.conv_Wz(x) + self.conv_Uz(h_prev))
        r = self.sigmoid(self.conv_Wr(x) + self.conv_Ur(h_prev))
        h_hat = torch.tanh(self.conv_W(x) + self.conv_U(r*h_prev))

        h = ((1-z) * h_prev) + (z * h_hat)
        return h