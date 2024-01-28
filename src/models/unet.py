import torch
import torch.nn as nn
import torch.nn.functional as F


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
        intermediate_output.append(x)

        for i, layer in enumerate(self.enc_blocks):
            x = layer(x)
            intermediate_output.append(x)

        return x, intermediate_output
    
class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, bilinear=False):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.layer_list = []
        for i in range(len(hidden_channels)-2):
            self.layer_list.append(Up(list(reversed(hidden_channels))[i],
                                    list(reversed(hidden_channels))[i+1],
                                    bilinear))
             
        self.layer_list.append(Down(hidden_channels[-2], hidden_channels[-1] // factor))
        self.dec_blocks = nn.ModuleList(self.layer_list)
        self.out_layer = OutConv(hidden_channels[0], out_channels)

    def forward(self, x, intermediate_output):
        intermediate_output = []

        for i, layer in enumerate(self.dec_blocks):
            x = layer(x, intermediate_output[-(i+1)])

        x = self.out_layer(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bilinear=False):
        """
        Class derived from U-Net with some modifications to give more flexibility to the layer size.
        Also renamed some variables to be more specific to our application.

        Arguments:
        - in_channels: int, number of input channels.
        - hidden_channels: list, number of channels in hidden layers.
        - out_channels: int, number of output channels.
        - bilinear: bool, whether to use linear interpolation during upscaling or transposes convolutions.
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hidden = hidden_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.encoder = Encoder(hidden_channels, bilinear)
        self.decoder = Decoder(hidden_channels, bilinear)

    def forward(self, x):
        x, intermediate_output = self.encoder(x)
        x = self.decoder(x, intermediate_output)
        return x