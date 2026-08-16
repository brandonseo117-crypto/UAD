from typing import Any

import torch
from torch.optim import Adam
import torch.nn as nn
from dataset import train_loader
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
import time
from monai.networks.nets.vitautoenc import ViTAutoEnc

class VitAE(nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 128)):
        super(VitAE, self).__init__()
        self.input_shape = input_shape
        self.model = ViTAutoEnc(
            img_size=(128, 128, 128),
            patch_size=(16, 16, 16),
            in_channels=1,
            out_channels=1,
            deconv_chns=16,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type="conv",
        )

    def forward(self, x):
        reconstructed_img, hidden_state = self.model(x) 
        return reconstructed_img, hidden_state

class VitDualDomainLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.4):
        super().__init__()
        self.alpha = alpha  # Feature alignment weight
        self.beta = beta    # Data-space Huber reconstruction weight
        self.huber = nn.HuberLoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, input_vol, target_vol, hidden_states):
        # 1. Voxel-space reconstruction loss
        l_data = self.huber(input_vol, target_vol)

        # 2. Feature-space cosine distance
        # Correctly matches e1 with d1 (base_dim) and e2 with d2 (base_dim*2)
        l_feat = 0.0
        if hidden_states is not None and len(hidden_states) > 1:
            # Compare earlier layers against the deepest layer
            ref_layer = hidden_states[-1]
            for state in hidden_states[:-1]:
                sim = self.cosine(state, ref_layer).mean()
                l_feat += (1.0 - sim)

        return self.alpha * l_feat + self.beta * l_data
    
if __name__ == "__main__":
    #loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VitAE().to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = VitDualDomainLoss()
    epochs = 5
    calculate_psnr = PeakSignalNoiseRatio(data_range=5.0).to(device)

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
            output, hidden_state = model(input_data)
            loss = criterion(output, input_data, hidden_state)
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
    plt.title('Training loss over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('figures/vit_convergence.svg', format='svg')
    plt.close()

    plt.figure(3)
    plt.plot(range(len(loss_history)), psnr_history, color='green', linewidth=2)
    plt.title('PSNR over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig('figures/vit_psnr.svg', format='svg')
    plt.close()