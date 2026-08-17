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
            in_shape=input_shape,
            out_channels=input_shape[0],
            latent_size=latent_dim,
            channels=(64, 128, 256),
            strides = (2,2,2),
            num_res_units=2
        )

    def forward(self, x):
        return self.vae(x)

    def anomaly_mapping(self, x):
        recon_x, _, _= self(x)
        return torch.abs(x - recon_x), recon_x 

class ELBOvae(torch.nn.Module):
    def __init__(self, beta=1e-3):
        super().__init__()
        self.beta = beta
        self.huber = torch.nn.HuberLoss()

    def forward(self, x, recon_x, mu, logvar):
        recon_loss = self.huber(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        return recon_loss + (self.beta * kld_loss)

if __name__ == "__main__":
    #loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VAEModel(input_shape=(1, 128, 128, 128), latent_dim=256).to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = ELBOvae()
    epochs = 5

    calculate_psnr = PeakSignalNoiseRatio(data_range=5.0).to(device)

    psnr_history = []
    vram_history = []
    loss_history =[]
    timestamps = []
    
    start_time = time.time()
    #pretend we have a dataloader called 'train_loader'
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, input_data in enumerate(train_loader):
            input_data = input_data.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad()
            (output, mu, logvar) = model(input_data)
            loss = criterion(input_data, output, mu, logvar)
            loss.backward()
            optimizer.step()

            peak_vram_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_mb = peak_vram_bytes / (1024**2)
            vram_history.append(peak_vram_mb)
            with torch.no_grad():
                psnr_val = calculate_psnr(output, input_data).item()
            psnr_history.append(psnr_val)
            loss_history.append(loss.item())

            timestamps.append(time.time() - start_time)
            
            running_loss += loss.item()

            

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'vae_weights.pth')

    plt.figure(1)
    plt.plot(timestamps, vram_history, color='blue', linewidth=2)
    plt.title('VRAM Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('VRAM Allocated (MB)')
    plt.grid(True)
    plt.savefig('figures/vae_vram_usage.svg', format='svg')
    plt.close()

    plt.figure(2)
    plt.plot(range(len(loss_history)), loss_history, color='blue', linewidth=2)
    plt.title('Training loss over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('figures/vae_convergence.svg', format='svg')
    plt.close()

    plt.figure(3)
    plt.plot(range(len(loss_history)), psnr_history, color='green', linewidth=2)
    plt.title('PSNR over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig('figures/vae_psnr.svg', format='svg')
    plt.close()