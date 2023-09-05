import torch
from torch import nn
from torchsummary import summary
from mobilenetv3 import SeModule, BtnkBlock, RevBtnkBlock

class FusedMBZBlock(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, activation, se, stride) -> None:
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=kernel_size, \
                               padding=kernel_size//2, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.activation1 = activation(inplace=True)

        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv2 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation2 = activation(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=kernel_size, groups=in_size, \
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, groups=in_size, \
                                   stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x
        x = self.activation1(self.bn1(self.conv1(x)))
        x - self.se(x)
        x = self.bn2(self.conv2(x))

        if self.skip is not None:
            skip = self.skip(skip)
        x = x + skip
        return self.activation2(x)

class RevFusedMBZBlock(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, activation, se, stride) -> None:
        super().__init__()
        self.stride = stride

        self.conv1 = nn.ConvTranspose2d(in_size, expand_size, kernel_size=kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.activation1 = activation(inplace=True)

        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv2 = nn.ConvTranspose2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation2 = activation(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_size, out_channels=in_size, kernel_size=kernel_size, groups=in_size, \
                                   stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        elif stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size, groups=in_size, \
                                   stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x
        x = self.activation1(self.bn1(self.conv1(x)))
        x - self.se(x)
        x = self.bn2(self.conv2(x))
        x = x + self.skip(skip)
        return self.activation2(x)



class EfficientNetV2(nn.Module):
    def __init__(self, in_chans, latent_dimension, act=nn.SiLU) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_chans, out_channels=24, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=24)
        self.act1 = act(inplace=True)

        self.stages = []
        input_channel = 24

        self.fusedMBConv_cfg = [  # Fused-MBConv in EffNet V2 S
            [1, 24, 2, 1, 0],  
            [4, 48, 4, 2, 0],
            [4, 64, 4, 2, 0],
        ]  # expansion ratio, channels num, layer num, stride, SE

        for cfg in self.fusedMBConv_cfg:
            output_channel = cfg[1]
            expand_size=input_channel*cfg[0]
            for i in range(cfg[2]):
                self.stages.append(
                    FusedMBZBlock(kernel_size=3, in_size=input_channel, expand_size=input_channel*cfg[0], \
                                 out_size=output_channel, activation=act, se=cfg[4], stride=(cfg[3] if i == 0 else 1))
                                 )
                input_channel = output_channel

        self.MBConv_cfg = [  # MBConv in EffNet V2 S
            [4, 128,  6, 2, 1],
            [6, 160,  9, 1, 1],
            [6, 256, 15, 2, 1],
        ] # expansion ratio, channels num, layer num, stride, SE

        for cfg in self.MBConv_cfg:
            output_channel = cfg[1]
            expand_size=input_channel*cfg[0]
            for i in range(cfg[2]):
                self.stages.append(
                    BtnkBlock(kernel_size=3, in_size=input_channel, expand_size=expand_size, \
                                 out_size=output_channel, activation=act, se=cfg[4], stride=(cfg[3] if i == 0 else 1))
                                 )
                input_channel = output_channel
        
        self.MBBlocks = nn.Sequential(*self.stages)

        self.conv2 = nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.act2 = act(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear3 = nn.Linear(256, 1024, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        self.act3 = act(inplace=True)
        self.linear4 = nn.Linear(1024, latent_dimension)

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
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.MBBlocks(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.avg_pool(x).flatten(1)
        x = self.act3(self.bn3(self.linear3(x)))

        return self.linear4(x)
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    effnet = EfficientNetV2(in_chans=2, latent_dimension=512)
    effnet.to(device)
    sample_input = (2, 224, 224)
    summary(effnet, input_size=sample_input)
    
