import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import ROI_BONES
from ..utils import ResLayer


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 bn=True,
                 act='relu',
                 padding_mode="zeros",
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels)

    def forward(self, x):
        return self.conv2(self.conv1(x))


@ROI_BONES.register_module()
class SimpleROINet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = DoubleConv2d(3, 64)
        self.conv2 = DoubleConv2d(64, 128)
        self.conv3 = DoubleConv2d(128, 256)
        self.conv4 = DoubleConv2d(256, 512)
        self.conv5 = DoubleConv2d(512, 512)
        self.roi_layer = DoubleConv2d(512, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        """
        ori:    8000x6000, window: 800x800, step: 400x400
        total:  20x15=300 rois
        2000x2000 -->
        input:  1280x960
        conv1:  640x480
        conv2:  320x240
        conv3:  160x120
        conv4:  80x60
        conv5:  40x30

        out:    10x10*2**5
        """
        conv = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for c in conv:
            x = c(x)
            x = self.maxpool(x)
        x = self.roi_layer(x)
        return x
