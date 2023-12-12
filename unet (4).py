import torch
from torch import nn
import torch.nn.functional as F

# class Normalize(nn.Module):
#     def __init__(self, mean, std):
#         super().__init__()
#         # Register mean and std as buffers
#         self.register_buffer('mean', mean.reshape(1, -1, 1, 1))
#         self.register_buffer('std', std.reshape(1, -1, 1, 1))
#
#     def forward(self, x):
#         return (x - self.mean) / self.std

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze, expand):
        super(FireModule, self).__init__()
        self.squeeze_conv = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze)
        self.expand1x1_conv = nn.Conv2d(squeeze, expand, kernel_size=1)
        self.expand3x3_conv = nn.Conv2d(squeeze, expand, kernel_size=3, padding=1)
        self.expand_bn = nn.BatchNorm2d(2 * expand)

    def forward(self, x):
        x = F.relu(self.squeeze_bn(self.squeeze_conv(x)))
        left = F.relu(self.expand1x1_conv(x))
        right = F.relu(self.expand3x3_conv(x))
        x = torch.cat([left, right], dim=1)
        x = self.expand_bn(x)
        return x

class FireModuleUpsample(nn.Module):
    def __init__(self, in_channels, squeeze, expand):
        super(FireModuleUpsample, self).__init__()
        self.squeeze_conv1x1 = nn.ConvTranspose2d(in_channels, squeeze, kernel_size=1)
        self.expand1x1_conv = nn.ConvTranspose2d(squeeze, expand, kernel_size=1)
        self.expand2x2_conv = nn.ConvTranspose2d(squeeze, expand, kernel_size=2, stride=2)
        self.expand_bn = nn.BatchNorm2d(2 * expand)

    def forward(self, x):
        x = F.relu(self.squeeze_conv1x1(x))
        left = F.relu(self.expand1x1_conv(x))
        right = F.relu(self.expand2x2_conv(x))
        x = torch.cat([left, right], dim=1)
        x = self.expand_bn(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, last=False):
        super().__init__()
        if last is False:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x should (N, C, H, W)
        return self.double_conv(x)


class UNet(nn.Module):
    """Unet inspired architecture.
    Using same convolutions, with output channels being equal to the number of classes. Adding instead of
    appending. Upsampling with MaxUnpooling instead of transpose convolutions.

    Attributes:
        in_channels: The number of input channels.
        out_channels: The number of output classes (including background).
        channel_list: A list representing the intermediate channels that we have. The bottleneck (bottom of the U) outputs 2 * the last channel.
    """

    def __init__(self, in_channels, out_channels, channel_list, means = None, stds = None):
        super().__init__()
        # if means is None:
        #     means = torch.tensor([0.0 for _ in range(in_channels)])
        # if stds is None:
        #     stds = torch.tensor([1.0 for _ in range(in_channels)])
        # self.normalize = Normalize(mean=means, std=stds)
        self.downs = nn.ModuleList()
        self.downs.append(DoubleConv(in_channels, 64, 64))
        self.downs.append(FireModule(64, 32, 64))
        self.downs.append(FireModule(128, 48,128))
        self.downs.append(FireModule(256,64,256 ))
        self.downs.append(FireModule(512,80,512 ))       


        self.ups = nn.ModuleList()
        self.ups.append(FireModuleUpsample(512, 80, 512))
        self.ups.append(FireModuleUpsample(256, 64, 256))
        self.ups.append(FireModuleUpsample(128, 48, 128))
        self.ups.append(FireModuleUpsample(64, 32, 64))
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        # x = self.normalize(x)
        pool_outs = []
        down_activations = []
        for down in self.downs:
            x = down(x)
            down_activations.append(x)

            # x, indices = self.pool(x)
            # pool_outs.append(indices)
        for index, up in enumerate(self.ups):
            x = torch.cat([x, down_activations[-1 - index]], dim=1)
            x = up(x)
        
        return x