from typing import Any, List, Optional

import torch
from torch import nn, Tensor


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


class BtnkBlock(nn.Module):
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
    

class RevBtnkBlock(nn.Module):
    """expand + depthwise + pointwise"""
    def __init__(self, kernel_size, in_size, expand_size, out_size, activation, se, stride) -> None:
        super().__init__()
        self.stride = stride

        self.conv1 = nn.ConvTranspose2d(in_size, expand_size, kernel_size=1, bias=False) # pointwise
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.activation1 = activation(inplace=True)

        self.conv2 = nn.ConvTranspose2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, \
                               padding=kernel_size//2, groups=expand_size, bias=False) # depthwise
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.activation2 = activation(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.ConvTranspose2d(expand_size, out_size, kernel_size=1, bias=False) # pointwise
        self.bn3 = nn.BatchNorm2d(out_size)
        self.activation3 = activation(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
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
    def __init__(self, in_chans, latent_dimension, act=nn.Hardswish):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            BtnkBlock(kernel_size=3, in_size=16, expand_size=16, out_size=16, activation=nn.ReLU, se=True, stride=2),
            BtnkBlock(kernel_size=3, in_size=16, expand_size=72, out_size=24, activation=nn.ReLU, se=False, stride=2),
            BtnkBlock(kernel_size=3, in_size=24, expand_size=88, out_size=24, activation=nn.ReLU, se=False, stride=1),
            BtnkBlock(kernel_size=5, in_size=24, expand_size=96, out_size=40, activation=act, se=True, stride=2),
            BtnkBlock(kernel_size=5, in_size=40, expand_size=240, out_size=40, activation=act, se=True, stride=1),
            BtnkBlock(kernel_size=5, in_size=40, expand_size=240, out_size=40, activation=act, se=True, stride=1),
            BtnkBlock(kernel_size=5, in_size=40, expand_size=120, out_size=48, activation=act, se=True, stride=1),
            BtnkBlock(kernel_size=5, in_size=48, expand_size=144, out_size=48, activation=act, se=True, stride=1),
            BtnkBlock(kernel_size=5, in_size=48, expand_size=288, out_size=96, activation=act, se=True, stride=2),
            BtnkBlock(kernel_size=5, in_size=96, expand_size=576, out_size=96, activation=act, se=True, stride=1),
            BtnkBlock(kernel_size=5, in_size=96, expand_size=576, out_size=96, activation=act, se=True, stride=1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1024, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1024, latent_dimension)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        print(x.shape)
        x = self.hs1(self.bn1(self.conv1(x)))
        print(x.shape)
        x = self.bneck(x)

        x = self.hs2(self.bn2(self.conv2(x)))
        x = self.ave_pool(x)
        print(x.shape)
        x = x.flatten(1)
        print(x.shape)
        x = self.drop(self.hs3(self.bn3(self.linear3(x))))

        return self.linear4(x)
    


class RevMobileNetV3(nn.Module):
    def __init__(self, latent_dimension, act=nn.Hardswish):
        super().__init__()

        self.linear_r4 = nn.Linear(latent_dimension, 1024)
        self.bn_r4 = nn.BatchNorm1d(1024)
        self.hs_r4 = act(inplace=True)

        self.drop = nn.Dropout(0.2)

        self.linear_r3 = nn.Linear(1024, 576, bias=False)
        self.hs_r3 = act(inplace=True)
        self.bn_r3 = nn.BatchNorm1d(576)
        
        self.upscale = nn.ConvTranspose2d(in_channels=576, out_channels=576, kernel_size=7, groups=576//16)


        self.conv_r2 = nn.ConvTranspose2d(576, 96, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_r2 = nn.BatchNorm2d(96)
        self.hs_r2 = act(inplace=True)

        self.bneck = nn.Sequential(  # reverted from MobileNetV3
            RevBtnkBlock(kernel_size=5, in_size=96, expand_size=576, out_size=96, activation=act, se=True, stride=1),
            RevBtnkBlock(kernel_size=5, in_size=96, expand_size=576, out_size=96, activation=act, se=True, stride=1),
            RevBtnkBlock(kernel_size=5, in_size=96, expand_size=288, out_size=48, activation=act, se=True, stride=2),
            RevBtnkBlock(kernel_size=5, in_size=48, expand_size=144, out_size=48, activation=act, se=True, stride=1),
            RevBtnkBlock(kernel_size=5, in_size=48, expand_size=120, out_size=40, activation=act, se=True, stride=1),
            RevBtnkBlock(kernel_size=5, in_size=40, expand_size=240, out_size=40, activation=act, se=True, stride=1),
            RevBtnkBlock(kernel_size=5, in_size=40, expand_size=240, out_size=40, activation=act, se=True, stride=1),
            RevBtnkBlock(kernel_size=5, in_size=40, expand_size=96, out_size=24, activation=act, se=True, stride=2),
            RevBtnkBlock(kernel_size=3, in_size=24, expand_size=88, out_size=24, activation=nn.ReLU, se=False, stride=1),
            RevBtnkBlock(kernel_size=3, in_size=24, expand_size=72, out_size=16, activation=nn.ReLU, se=False, stride=2),
            RevBtnkBlock(kernel_size=3, in_size=16, expand_size=16, out_size=16, activation=nn.ReLU, se=True, stride=2), 
        )

        self.conv_r1 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.hs_r4(self.bn_r4(self.linear_r4(x)))
        x = self.drop(x)
        x = self.hs_r3(self.bn_r3(self.linear_r3(x)))
        x = torch.unsqueeze(torch.unsqueeze(x, dim=-1), dim=-1)
        print(x.shape)
        x = self.upscale(x)
        x = self.hs_r2(self.bn_r2(self.conv_r2(x)))
        x = self.bneck(x)
        x = self.conv_r1(x)

        return x