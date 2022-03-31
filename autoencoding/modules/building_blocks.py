import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d_CBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv1d_CBnReLU, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class Conv2d_CBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d_CBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class ConvTransposed1d_CBnReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
    ):
        super(ConvTransposed1d_CBnReLU, self).__init__()
        self.convt = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.convt(x)))


class ConvTransposed2d_CBnReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
    ):
        super(ConvTransposed2d_CBnReLU, self).__init__()
        self.convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.convt(x)))


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, width=None):
        super(ResidualBlock2d, self).__init__()

        if width is None:
            width = in_channels

        self.branch = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(width, in_channels, 1),
        )

    def forward(self, x):
        return self.branch(x) + x
