import torch
import torch.nn as nn
from convnext import ConvNeXt, RevConvNext
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNextAE(nn.Module):
    def __init__(self, latent_size):
        super(ConvNextAE, self).__init__()
        self.latent_size = latent_size
        self.depth = [3, 3, 9, 3]
        self.dims = [96, 192, 384, 768]
        self.fc_latent2 = self.dims[-1]

        self.encoder = ConvNeXt(in_chans=2, depths=self.depth, dims=self.dims)
        self.decoder = RevConvNext(
            in_chans=768, depths=self.depth, dims=self.dims[::-1]
        )
        # only flip the channel number for blocks
        self.pool_fc = nn.Linear(768, self.latent_size)
        self.de_fc1 = nn.Linear(self.latent_size, self.fc_latent2)

    def encode(self, img_label):
        t = self.encoder(img_label)
        t = self.pool_fc(t)
        return t

    def decode(self, z):
        t = self.de_fc1(z)
        t = t.unsqueeze(-1).unsqueeze(-1)
        return self.decoder(t)

    def forward(self, img_label):
        z = self.encode(img_label)
        return self.decode(z)


if __name__ == "__main__":
    convnext_ae = ConvNextAE(512)
    convnext_ae.to(device)
    sample_input = (2, 224, 224)
    summary(convnext_ae, input_size=sample_input)
