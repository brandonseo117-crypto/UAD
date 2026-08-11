import torch
import torch.nn as nn

class ViTAutoencoder(nn.Module):
    def __init__(self, in_channels=1, img_size=(128,128,128), patch_size=(16, 16, 16), embed_dim=768, num_heads=12, depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.depth = depth

        self.num_patches = (
            (img_size[0] // patch_size[0]) * 
            (img_size[1] // patch_size[1]) * 
            (img_size[2] // patch_size[2])
        )
        self.patchification = nn.Conv3d(
                                   in_channels=self.in_channels, 
                                   out_channels=self.embed_dim, 
                                   kernel_size=(self.patch_size), 
                                   stride=(self.patch_size)
                                )
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim*4,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers=depth)
    def patching_input(self, x):
        conv_out = self.patchification(x)
        return conv_out.flatten(2, -1).transpose(1,2)

    def forward(self, x):
        tokens = self.patching_input(x)
        tokens = tokens + self.pos_embedding
        encoded_tokens = self.transformer_encoder(tokens)


        return encoded_tokens
        
    

    


