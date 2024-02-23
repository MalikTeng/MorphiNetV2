import torch
import torch.nn as nn

from collections.abc import Callable
from functools import partial
from typing import Any, Union

from monai.networks.layers.factories import Conv, Norm


__all__ = [
    "ResNetBlock", "ResNetBottleneck"
    ]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class ResBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         norm, act, 
#         num_layers,
#         ):
#         super(ResBlock, self).__init__()
#         assert out_channels is not None

#         self.blocks = []
#         for i in range(num_layers):
#             if i == 0:
#                 conv_layer = ConvLayer(
#                     in_channels, out_channels, 3, 1, 1,
#                     norm, act
#                 )
#             else:
#                 conv_layer = ConvLayer(
#                     out_channels, out_channels, 3, 1, 1,
#                     norm, act
#                 )
#             self.blocks.append(conv_layer)

#         self.blocks = nn.Sequential(*self.blocks)

#     def forward(self, x):
#         x_ = self.blocks(x)
#         x = x + x_

#         return x, x_


# class ConvLayer(nn.Module):
#     def __init__(
#         self,
#         in_channels, out_channels,
#         kernel_size, stride, padding,
#         norm, act, 
#     ):
#         super(ConvLayer, self).__init__()

#         self.conv_layer = nn.Conv3d(
#             in_channels, out_channels, 
#             bias=True if norm == 'batch' else False,
#             kernel_size=kernel_size, stride=stride, padding=padding
#         )

#         if norm == 'batch':
#             self.norm = nn.BatchNorm3d(out_channels)
#         elif norm == 'instance':
#             self.norm = nn.InstanceNorm3d(out_channels)
#         else:
#             self.norm = None
        
#         if act == 'relu':
#             self.act = nn.LeakyReLU(1e-2, inplace=True)
#         elif act == 'tanh':
#             self.act = nn.Tanh()
#         elif act == 'sigmoid':
#             self.act = nn.Sigmoid()
#         else:
#             self.act = None

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, 1e-2, mode='fan_out', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv_layer(x)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.act is not None:
#             x = self.act(x)

#         return x

