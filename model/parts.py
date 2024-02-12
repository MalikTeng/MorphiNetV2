import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ConvLayer", "ResBlock"
    ]

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm, act, 
        num_layers,
        ):
        super(ResBlock, self).__init__()
        assert out_channels is not None

        self.blocks = []
        for i in range(num_layers):
            if i == 0:
                conv_layer = ConvLayer(
                    in_channels, out_channels, 3, 1, 1,
                    norm, act
                )
            else:
                conv_layer = ConvLayer(
                    out_channels, out_channels, 3, 1, 1,
                    norm, act
                )
            self.blocks.append(conv_layer)

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x_ = self.blocks(x)
        x = x + x_

        return x, x_


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size, stride, padding,
        norm, act, 
    ):
        super(ConvLayer, self).__init__()

        self.conv_layer = nn.Conv3d(
            in_channels, out_channels, 
            bias=True if norm == 'batch' else False,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

        if norm == 'batch':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = None
        
        if act == 'relu':
            self.act = nn.LeakyReLU(1e-2, inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, 1e-2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x

