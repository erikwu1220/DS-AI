import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, bilinear=False):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bilinear = bilinear
        
        self.in_layer = DoubleConv(in_channels, hidden_channels[0])

        factor = 2 if bilinear else 1
        self.layer_list = []
        for i in range(len(hidden_channels)-2):
            self.layer_list.append(Down(hidden_channels[i], 
                                        hidden_channels[i+1]))
             
        self.layer_list.append(Down(hidden_channels[-2], hidden_channels[-1] // factor))
        self.enc_blocks = nn.ModuleList(self.layer_list)

    def forward(self, x):
        intermediate_output = []

        x = self.in_layer(x)

        for i, layer in enumerate(self.enc_blocks):
            intermediate_output.append(x)
            x = layer(x)

        return x, intermediate_output
    
class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, bilinear=False):
        super(Decoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.layer_list = []
        for i in range(len(hidden_channels)-2):
            self.layer_list.append(Up(list(reversed(hidden_channels))[i],
                                    list(reversed(hidden_channels))[i+1]//factor,
                                    bilinear))
            
        self.layer_list.append(Up(hidden_channels[1],
                                  hidden_channels[0],
                                  bilinear))
        
        self.dec_blocks = nn.ModuleList(self.layer_list)
        self.out_layer = OutConv(hidden_channels[0], out_channels)

    def forward(self, x, intermediate_output):

        for i, layer in enumerate(self.dec_blocks):
            x = layer(x, intermediate_output[-(i+1)])

        x = self.out_layer(x)
        return x


class UNet_mask(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, distance = None, bilinear=False):
        """
        Class derived from U-Net with some modifications to give more flexibility to the layer size.
        Also renamed some variables to be more specific to our application.

        Arguments:
        - in_channels: int, number of input channels.
        - hidden_channels: list, number of channels in hidden layers.
        - out_channels: int, number of output channels.
        - bilinear: bool, whether to use linear interpolation during upscaling or transposes convolutions.
        """
        super(UNet_mask, self).__init__()
        self.in_channels = in_channels
        self.hidden = hidden_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.encoder = Encoder(in_channels, hidden_channels, bilinear)
        self.decoder = Decoder(hidden_channels, out_channels, bilinear)

        if not distance:
            self.distance = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            self.distance = distance

    def forward(self, x):
        inputs = x
        x, intermediate_output = self.encoder(x)
        x = self.decoder(x, intermediate_output)

        output = torch.empty_like(x) # .to(matrix.device)

        # Loop through the batch
        for i in range(x.shape[0]):

            if isinstance(self.distance, torch.nn.parameter.Parameter):
                distance_matrix = self.distance_to_nonzero(inputs[i,1,:,:])
                distance_matrix[distance_matrix == 0] = 1
                output[i] = x[i] * distance_matrix ** (-self.distance * 10)
            
            else:
                # Generate a mask
                mask = self.get_mask(inputs[i,1,:,:])

                # Apply mask to the output
                output[i] = x[i] * mask

        return output

    def distance_to_nonzero(self, matrix):

        rows, cols = matrix.shape

        # Create an array with indices corresponding to each element in the input matrix
        indices = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")

        # Find the indices where the matrix is nonzero
        nonzero_indices = torch.nonzero(matrix, as_tuple=True)

        # Calculate distances using broadcasting
        distances = torch.abs(indices[0].to(matrix.device)[:, :, None, None] - nonzero_indices[0]) + \
                    torch.abs(indices[1].to(matrix.device)[:, :, None, None] - nonzero_indices[1])

        # Find the minimum distance for each element and set the result matrix
        result_matrix, _ = torch.min(distances, dim=-1)

        return result_matrix.squeeze().to(matrix.device)

    def get_mask(self, matrix, threshold = 1e-4):
        distance_matrix = self.distance_to_nonzero(matrix)
        return torch.lt(distance_matrix, self.distance + threshold)