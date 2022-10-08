from config import *
import torch.nn as nn


class SeperableConv2d(nn.Module):

    # ***Figure 4. An “extreme” version of our Inception module,
    # with one spatial convolution per output channel of the 1x1
    # convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            bias=False,
            **kwargs
        )
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# noinspection PyTypeChecker
class EntryFlow(nn.Module):

    def __init__(self, num_init_channels):
        super().__init__()
        self.num_init_channels = num_init_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_init_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )
        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )
        # no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )
        # no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        # print("EntryFlow_input.shape", x.shape)
        x = self.conv1(x)
        # print("conv1.shape", x.shape)
        x = self.conv2(x)
        # print("conv2.shape", x.shape)
        residual = self.conv3_residual(x)
        # print("conv3_residual.shape", residual.shape)
        shortcut = self.conv3_shortcut(x)
        # print("conv3_shortcut.shape", shortcut.shape)
        x = residual + shortcut
        # print("residual + shortcut.shape", x.shape)
        residual = self.conv4_residual(x)
        # print("conv4_residual.shape", residual.shape)
        shortcut = self.conv4_shortcut(x)
        # print("conv4_shortcut.shape", shortcut.shape)
        x = residual + shortcut
        # print("residual + shortcut.shape", x.shape)
        residual = self.conv5_residual(x)
        # print("conv5_residual.shape", residual.shape)
        shortcut = self.conv5_shortcut(x)
        # print("conv5_shortcut.shape", shortcut.shape)
        x = residual + shortcut
        # print("residual + shortcut.shape", x.shape)
        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        # print("MiddleFLowBlock_input.shape", x.shape)
        residual = self.conv1(x)
        # print("conv1.shape", residual.shape)
        residual = self.conv2(residual)
        # print("conv2.shape", residual.shape)
        residual = self.conv3(residual)
        # print("conv3.shape", residual.shape)
        shortcut = self.shortcut(x)
        # print("shortcut.shape", shortcut.shape)
        temp = shortcut + residual
        # print("shortcut + residual.shape", temp.shape)
        return shortcut + residual


class MiddleFlow(nn.Module):
    def __init__(self, block):
        super().__init__()
        # """then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8)

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())
        return nn.Sequential(*flows)

    def forward(self, x):
        x = self.middel_block(x)
        return x


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )
        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # print("ExitFLow_input.shape", x.shape)
        shortcut = self.shortcut(x)
        # print("shortcut.shape", shortcut.shape)
        residual = self.residual(x)
        # print("residual.shape", residual.shape)
        output = shortcut + residual
        # print("shortcut + residual.shape", output.shape)
        output = self.conv(output)
        # print("conv.shape", output.shape)
        output = self.avgpool(output)
        # print("avgpool.shape", output.shape)
        return output


class Xception(nn.Module):

    def __init__(self, num_init_channels, block, dropout, num_class=7):
        super().__init__()
        self.num_init_channels = num_init_channels
        self.entry_flow = EntryFlow(num_init_channels)
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()
        self.fc = nn.Linear(2048, num_class)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(2048, num_class)
        )

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)    # dropout
        # x = self.fc(x)    # no dropout
        return x


def x_ception(dropout):
    return Xception(dropout=dropout, block=MiddleFLowBlock, num_init_channels=data_config.input_channel, num_class=data_config.num_class)

