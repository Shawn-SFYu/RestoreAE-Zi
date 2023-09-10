import torch
from torch import nn
from torchsummary import summary
from .mobilenetv3 import MobileNetV3, InvMobileNetV3

class MobileNetAE(nn.Module):
    def __init__(self, latent_dimension) -> None:
        super().__init__()
        self.encoder = MobileNetV3(in_chans=2, latent_dimension=latent_dimension)
        self.decoder = InvMobileNetV3(latent_dimension=latent_dimension)

    def forward(self, img_label):
        z = self.encoder(img_label)
        x = self.decoder(z)
        return x
    
if __name__ == "__main__":
    mobilenet_ae = MobileNetAE(512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mobilenet_ae.to(device)
    sample_input = (2, 224, 224)
    summary(mobilenet_ae, input_size=sample_input)
