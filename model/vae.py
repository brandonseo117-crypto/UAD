import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from monai.networks.nets import VarAutoEncoder
from dataset import train_loader, val_loader
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
        recon_x, _, _, _ = self(x)
        return torch.abs(x - recon_x), recon_x 

class ELBOvae(torch.nn.Module):
    def __init__(self, beta=1e-3):
        super().__init__()
        self.beta = beta
        self.huber = torch.nn.HuberLoss(reduction="mean")

    def forward(self, x, recon_x, mu, logvar):
        recon_loss = self.huber(recon_x, x)
        kld_elements = 1 + logvar - mu**2 - logvar.exp()
        kld_loss = torch.mean(-0.5 * torch.sum(kld_elements, dim=1))
        return recon_loss + (self.beta * kld_loss)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VAEModel(input_shape=(1, 128, 128, 128), latent_dim=256).to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = ELBOvae()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, min_lr=1e-6)
    epochs = 15

    calculate_psnr = PeakSignalNoiseRatio(data_range=5.0).to(device)

    vram_history = []
    loss_history = []
    val_loss_history = []
    psnr_history = []
    val_psnr_history = []

    best_val_psnr = -float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        torch.cuda.reset_peak_memory_stats(device)
        for idx, input_data in enumerate(train_loader):
            input_data = input_data.squeeze(dim=0).to(device)
            optimizer.zero_grad()
            output, mu, logvar, _ = model(input_data)
            loss = criterion(input_data, output, mu, logvar)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                psnr_val = calculate_psnr(output, input_data).item()
                running_loss += loss.item()
                running_psnr += psnr_val
                peak_vram_bytes = torch.cuda.max_memory_allocated(device)
                peak_vram_mb = peak_vram_bytes / (1024 ** 2)
                current_lr = optimizer.param_groups[0]['lr']

                if idx % 10 == 0:
                    print(f'Batch {idx} Peak VRAM: {peak_vram_mb:.1f}MB Loss: {loss.item():.4f} PSNR: {psnr_val:.2f} current lr: {current_lr}')

        model.eval()
        val_running_loss = 0.0
        val_running_psnr = 0.0
        with torch.no_grad():
            for idx, input_data in enumerate(val_loader):
                input_data = input_data.squeeze(dim=0).to(device)
                output, mu, logvar, _ = model(input_data)
                loss = criterion(input_data, output, mu, logvar)
                psnr_val = calculate_psnr(output, input_data).item()
                val_running_loss += loss.item()
                val_running_psnr += psnr_val

        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        avg_val_psnr = val_running_psnr / len(val_loader)
        val_psnr_history.append(avg_val_psnr)
        avg_psnr = running_psnr / len(train_loader)
        psnr_history.append(avg_psnr)
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        epoch_max_vram_bytes = torch.cuda.max_memory_allocated(device)
        epoch_max_vram_mb = epoch_max_vram_bytes / (1024 ** 2)
        vram_history.append(epoch_max_vram_mb)
        print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")

        scheduler.step(avg_val_psnr)
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(model.state_dict(), 'weights/best_vae_weights.pth')

    torch.save(model.state_dict(), 'weights/end_vae_weights.pth')

    epoch_axis = [num+1 for num in range(15)]

    plt.figure(1)
    plt.plot(epoch_axis, loss_history, color='blue', linewidth=2)
    plt.plot(epoch_axis, val_loss_history, color='orange', linestyle='--', linewidth=2)
    plt.title('Training loss vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.savefig('figures/vae_convergence.svg', format='svg')
    plt.close()

    plt.figure(2)
    plt.plot(epoch_axis, psnr_history, color='green', linewidth=2)
    plt.plot(epoch_axis, val_psnr_history, color='red', linestyle='--', linewidth=2)
    plt.title('PSNR vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.savefig('figures/vae_psnr.svg', format='svg')
    plt.close()

    plt.figure(3)
    plt.plot(epoch_axis, vram_history, color='purple', linewidth=2)
    plt.title('VRAM vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('VRAM usage (MB)')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.savefig('figures/vae_vram.svg', format='svg')
    plt.close()