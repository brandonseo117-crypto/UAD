import torch
from torch.optim import Adam
import torch.nn as nn
from dataset import train_loader
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import time

class ViTVAE(nn.Module):
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

class ELBOvit(nn.Module):
    def __init__(self, beta=1e-3):
        super().__init__()
        self.beta = beta

    def forward(self, x, recon_x, mu, logvar):
        recon_loss = torch.nn.functional.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=(1,2)))
        return recon_loss + (self.beta * kld_loss)

if __name__ == "__main__":
    #loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ViTVAE().to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = ELBOvit()
    epochs = 5
    calculate_psnr = PeakSignalNoiseRatio(data_range=None).to(device)

    psnr_history = []
    vram_history = []
    loss_history = []
    timestamps = []

    start_time = time.time()
    #pretend we have a dataloader called 'train_loader'
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            input_data = input_data.to(device)
            output, mu, logvar = model(input_data)
            loss = criterion(input_data, output, mu, logvar)
            loss.backward()
            optimizer.step()
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            peak_vram_mb = peak_vram_bytes / (1024**2)
            vram_history.append(peak_vram_mb)
            loss_history.append(loss.item())
            psnr_history.append(calculate_psnr(output.detach(), input_data).item())
            timestamps.append(time.time() - start_time)
            
            print(peak_vram_mb)
            running_loss += loss.item()
            torch.cuda.reset_peak_memory_stats()


        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'vit_weights.pth')

    plt.figure(1)
    plt.plot(timestamps, vram_history, color='blue', linewidth=2)
    plt.title('VRAM Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('VRAM Allocated (MB)')
    plt.grid(True)
    plt.savefig('figures/vit_vram_usage.svg', format='svg')
    plt.close()

    plt.figure(2)
    plt.plot(range(len(loss_history)), loss_history, color='blue', linewidth=2)
    plt.plot(range(len(loss_history)), psnr_history, color='green', linewidth=2)
    plt.title('Training loss and PSNR over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('figures/vit_convergence.svg', format='svg')
    plt.close()