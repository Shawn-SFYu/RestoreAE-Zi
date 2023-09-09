import torch
import timm
import numpy as np
from torch import nn
from torchsummary import summary

# from einops import repeat, rearrange
# from einops.layers.torch import Rearrange

# from timm.models.layers import trunc_normal_
from torch.nn.init import trunc_normal_
from timm.models.vision_transformer import Block

'''
def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        return patches, forward_indexes, backward_indexes
'''

class ViT_Encoder(nn.Module):
    def __init__(self,
                 in_channels=2,
                 image_size=224,
                 patch_size=16,
                 emb_dim=768,
                 num_layer=12,
                 num_head=12,
                 ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))

        self.patchify = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img).contiguous()
        patches = patches.permute(2, 3, 0, 1) # 'b c h w -> (h w) b c'
        patches = patches.flatten(start_dim=0, end_dim=1)
        patches = patches + self.pos_embedding
        # cls_token 1, 1, em_dim -> 1, batch, em_dim
        # cat cls_token to patches_embed (h w)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = patches.permute(1, 0, 2) # 'token batch channel -> b t c'
        features = self.layer_norm(self.transformer(patches))
        features = features.permute(1, 0, 2) #  'b t c -> t b c'

        return features

class ViT_Decoder(nn.Module):
    def __init__(self,
                 out_channel=1,
                 image_size=224,
                 patch_size=16,
                 emb_dim=768,
                 num_layer=2,
                 num_head=12,
                 ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = image_size // patch_size
        self.pos_embedding = nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = nn.Linear(emb_dim, out_channel * patch_size ** 2)
        self.init_weight()
    
    def patch2img(self, patch: torch.Tensor, patch_size, patch_num):
        batch = patch.shape[1] # (t b img_channel * patch size ^ 2)
        patch = patch.contiguous().view(patch_num, patch_num, batch, -1, patch_size, patch_size) 
        # (patch_num, patch_num, batch, img_channel, patch_size, patch_size)
        patch = patch.permute(2, 3, 0, 4, 1, 5).contiguous()
        return patch.view(batch, -1, patch_num*patch_size, patch_num*patch_size)

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features):
        features = features + self.pos_embedding
        features = features.permute(1, 0, 2) # ('token b c -> b t c')
        features = self.transformer(features)
        features = features.permute(1, 0, 2) # ('b t c -> t b c')
        features = features[1:] # remove global feature, the feature is kept in this implemation

        patches = self.head(features) # (t b c) x (c img_channel * patch size ^ 2)
        img = self.patch2img(patches, patch_size=self.patch_size, patch_num=self.patch_num)

        return img 


class ViT_AE(nn.Module):
    def __init__(self,
                 in_channels=2,
                 image_size=224,
                 patch_size=16,
                 encoder_emb_dim=768,
                 decoder_emb_dim=768,
                 encoder_layer=12,
                 encoder_head=12,
                 decoder_layer=12,
                 decoder_head=12,
                 ) -> None:
        super().__init__()

        self.encoder = ViT_Encoder(in_channels, image_size, patch_size, encoder_emb_dim, encoder_layer, encoder_head)
        self.decoder = ViT_Decoder(image_size, patch_size, decoder_emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features = self.encoder(img)
        predicted_img = self.decoder(features)
        return predicted_img
    

if __name__ == "__main__":
    vit_ae = ViT_AE()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_ae.to(device)
    sample_input = (2, 224, 224)
    summary(vit_ae, input_size=sample_input)
