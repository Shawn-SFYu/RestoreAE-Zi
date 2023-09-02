from typing import Any, List, Optional

import torch
from torch import nn, Tensor


"""
class InvertedResidualConfig():
    def __init__(self,
                 input_channels: int,
                 kernel: int,
                 expanded_channels: int,
                 out_channels: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 dilation: int,
                 width_mult: float) -> None:
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels():
        return None


class InvertedResidual(nn.Module):
    def __init__(self, 
                 cnfg: InvertedResidualConfig,
                 norm_layer,
                 se_layer) -> None:
        super().__init__()
        if not (1 <= cnfg.stride <= 2):
            raise ValueError("illegal stride value")
        
        self.use_res_connect = cnfg.stride == 1 and cnfg.input_channels == cnfg.out_channels

        layers: List[nn.Module] = []
        activation = nn.Hardswish if cnfg.use_hs else nn.ReLU

        if cnfg.expanded_channels != cnfg.input_channels:
            layers.append(

            )


def _make_divisible(v: float, divisor: int, min_value: Optional[int]):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
"""

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    """expand + depthwise + pointwise"""
    def __init__(self, kernel_size, in_size, expand_size, out_size, activation, se, stride) -> None:
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False) # pointwise
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.activation1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, \
                               padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.activation2 = activation(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False) # pointwise
        self.bn3 = nn.BatchNorm2d(out_size)
        self.activation3 = activation(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        x = self.activation1(self.bn1(self.conv1(x)))
        x = self.activation2(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = self.bn3(self.conv3(x))

        if self.skip is not None:
            skip = self.skip(skip)
        x = skip + x
        return self.activation3(x)        
    

class MobileNetV3(nn.Module):
    def __init__(self, latent_dimension, act=nn.Hardswish):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(kernel_size=3, in_size=16, expand_size=16, out_size=16, activation=nn.ReLU, se=True, stride=2),
            Block(kernel_size=3, in_size=16, expand_size=72, out_size=24, activation=nn.ReLU, se=False, stride=2),
            Block(kernel_size=3, in_size=24, expand_size=88, out_size=24, activation=nn.ReLU, se=False, stride=1),
            Block(kernel_size=5, in_size=24, expand_size=96, out_size=40, activation=act, se=True, stride=2),
            Block(kernel_size=5, in_size=40, expand_size=240, out_size=40, activation=act, se=True, stride=1),
            Block(kernel_size=5, in_size=40, expand_size=240, out_size=40, activation=act, se=True, stride=1),
            Block(kernel_size=5, in_size=40, expand_size=120, out_size=48, activation=act, se=True, stride=1),
            Block(kernel_size=5, in_size=48, expand_size=144, out_size=48, activation=act, se=True, stride=1),
            Block(kernel_size=5, in_size=48, expand_size=288, out_size=96, activation=act, se=True, stride=2),
            Block(kernel_size=5, in_size=96, expand_size=576, out_size=96, activation=act, se=True, stride=1),
            Block(kernel_size=5, in_size=96, expand_size=576, out_size=96, activation=act, se=True, stride=1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1280, latent_dimension)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.ave_pool(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.linear4(out)