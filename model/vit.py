import torch
from torch.optim import Adam
import torch.nn as nn

class ViTAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, img_size=(128,128,128), patch_size=(16, 16, 16), embed_dim=768, num_heads=12, depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.depth = depth
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
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

        self.fc_mu = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_logvar = nn.Linear(self.embed_dim, self.embed_dim)
        self.decoder_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.decoder_conv = nn.ConvTranspose3d(
            in_channels=self.embed_dim,
            out_channels=self.in_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
    
    def patching_input(self, x):
        conv_out = self.patchification(x)
        return conv_out.flatten(2, -1).transpose(1,2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, x, beta=1e-3):
            recon_x, mu, logvar = self.forward(x)
            recon_loss = torch.nn.functional.mse_loss(recon_x, x)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=(1,2)))
            return recon_loss + (beta * kld_loss)

    def forward(self, x):
        tokens = self.patching_input(x)
        tokens = tokens + self.pos_embedding
        encoded_tokens = self.transformer_encoder(tokens)
        mu = self.fc_mu(encoded_tokens)
        logvar = self.fc_logvar(encoded_tokens)
        z = self.reparameterize(mu, logvar)
        dec_tokens = self.decoder_linear(z)
        dec_tokens = dec_tokens.transpose(1,2)
        dec_grid = dec_tokens.view(x.shape[0], self.embed_dim, self.grid_size[0], self.grid_size[1], self.grid_size[2])
        reconstructed_img = self.decoder_conv(dec_grid)
        return reconstructed_img, mu, logvar

#loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ViTAutoEncoder().to(device)
model.train()
optimizer = Adam(model.parameters(), lr=1e-3)
epochs = 5

#pretend we have a dataloader called 'train_loader'
for epoch in range(epochs):
    running_loss = 0.0
    for idx, input in enumerate(train_loader):
        optimizer.zero_grad()
        input = input.to(device)
        loss = model.compute_loss(input)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")