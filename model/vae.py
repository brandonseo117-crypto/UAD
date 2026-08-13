import torch
from torch.optim import Adam
from monai.networks.nets import VarAutoEncoder
from dataset import train_loader
from torchmetrics.image import PeakSignalNoiseRatio
import time
import matplotlib.pyplot as plt

class VAEModel(torch.nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 128), latent_dim=256):
        super(VAEModel, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.vae = VarAutoEncoder(
            spatial_dims=3,
            in_channels=input_shape[0],
            out_channels=input_shape[0],
            latent_size=latent_dim,
            channels=(64, 128, 256),
            strides = (2,2,2),
            num_res_units=2
        )

    def forward(self, x):
        return self.vae(x)

    def compute_loss(self, x, beta=1e-3):
        recon_x, mu, logvar = self.forward(x)
        recon_loss = torch.nn.functional.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=(1,2)))
        return recon_loss + (beta * kld_loss)

    def anomaly_mapping(self, x):
        recon_x, _, _ = self.forward(x)
        return torch.abs(x - recon_x), recon_x 

if __name__ == "__main__":
    #loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VAEModel(input_shape=(1, 128, 128, 128), latent_dim=256).to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 5
    vram_history = []
    timestamps = []
    start_time = time.time()
    #pretend we have a dataloader called 'train_loader'
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            input_data = input_data.to(device)
            loss = model.compute_loss(input_data)
            loss.backward()
            optimizer.step()

            peak_vram_bytes = torch.cuda.max_memory_allocated()
            peak_vram_mb = peak_vram_bytes / (1024**2)
            vram_history.append(peak_vram_mb)
            timestamps.append(time.time() - start_time)
            
            running_loss += loss.item()

            torch.cuda.reset_peak_memory_stats()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'vae_weights.pth')

    plt.plot(timestamps, vram_history, color='blue', linewidth=2)
    plt.title('VRAM Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('VRAM Allocated (MB)')
    plt.grid(True)
    plt.show()