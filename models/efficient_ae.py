import torch
from torch import nn
from torchsummary import summary
from .efficientnetv2 import EfficientNetV2, InvEfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EfficientNetAE(nn.Module):
    def __init__(self, latent_dimension) -> None:
        super().__init__()
        self.encoder = EfficientNetV2(in_chans=2, latent_dimension=latent_dimension)
        self.decoder = InvEfficientNet(latent_dimension=latent_dimension)

    def forward(self, img_label):
        z = self.encoder(img_label)
        x = self.decoder(z)
        return x
    
if __name__ == "__main__":
    mobilenet_ae = EfficientNetAE(512)
    mobilenet_ae.to(device)
    sample_input = (2, 224, 224)
    summary(mobilenet_ae, input_size=sample_input)