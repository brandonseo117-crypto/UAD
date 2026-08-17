import torch
import torch.nn as nn
from einops import rearrange
from torch.optim import Adam
from dataset import train_loader
from mamba_ssm import Mamba
from torchmetrics.image import PeakSignalNoiseRatio
import time
import matplotlib.pyplot as plt

class GatedSpatialConv3d(nn.Module): #good
    def __init__(self, channels):
        super().__init__()
        self.conv_3x3 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.SiLU()
        )

        self.conv_1x1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.SiLU()
        )

        self.fuse = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        gated = self.conv_3x3(x) * self.conv_1x1(x)
        return x + self.fuse(gated)

# ToM, TSMamba, and U-shape ae still needed
# loss fn


class TriOrientatedMamba(nn.Module):
    """
    ToM Module (SegMamba architecture): Captures 3D spatial dependencies 
    by scanning through Forward, Reverse, and Inter-slice orientations.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required. Install via `pip install mamba-ssm` "
                "in a CUDA-enabled environment."
            )
            
        # Distinct Mamba instances for each orientation
        self.mamba_f = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_r = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_s = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W  # Total sequence length

        # --- Orientation 1: Forward Direction (zf) ---
        # (B, C, D, H, W) -> (B, L, C)
        x_f = rearrange(x, 'b c d h w -> b (d h w) c')
        y_f = self.mamba_f(x_f)
        y_f = rearrange(y_f, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        # --- Orientation 2: Reverse Direction (zr) ---
        # Correctly flip along dimension 1 (sequence axis L)
        x_r = torch.flip(x_f, dims=[1]).contiguous()
        y_r = self.mamba_r(x_r)
        y_r = torch.flip(y_r, dims=[1]).contiguous()  # Flip back to restore original alignment
        y_r = rearrange(y_r, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)

        # --- Orientation 3: Inter-slice Direction (zs) ---
        # Transpose Depth (D) and Width (W)
        x_s = rearrange(x, 'b c d h w -> b (w h d) c')
        y_s = self.mamba_s(x_s)
        y_s = rearrange(y_s, 'b (w h d) c -> b c w h d', d=D, h=H, w=W)
        y_s = rearrange(y_s, 'b c w h d -> b c d h w')

        # Fusion: Element-wise sum across all 3 spatial scans
        return y_f + y_r + y_s


class TSMambaBlock(nn.Module): # good
    def __init__(self, dim):
        super().__init__()
        self.gsc = GatedSpatialConv3d(dim)
        self.tom = TriOrientatedMamba(d_model=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # 1. Local Spatial Gating [3]
        x = self.gsc(x)

        # 2. Global Tri-orientated Scanning [3]
        B, C, D, H, W = x.shape
        res = x
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm1(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        x = self.tom(x) + res

        # 3. Feature Enrichment (MLP) [3]
        res = x
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.norm2(x)
        x = self.mlp(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        return x + res


class MambaUAD(nn.Module): #Good
    def __init__(self, in_channels=1, base_dim=48):
        super().__init__()
        # Stem: 7x7x7 Depth-wise Conv, Stride 2 [10]
        self.stem = nn.Conv3d(in_channels, base_dim,
                              kernel_size=7, stride=2, padding=3)

        # Encoder Layers [11]
        self.enc1 = TSMambaBlock(base_dim)
        self.down1 = nn.Conv3d(base_dim, base_dim*2, kernel_size=2, stride=2)

        self.enc2 = TSMambaBlock(base_dim*2)
        self.down2 = nn.Conv3d(base_dim*2, base_dim*4, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = TSMambaBlock(base_dim*4)

        # Decoder Layers (Symmetric) [9]
        self.up2 = nn.ConvTranspose3d(
            base_dim*4, base_dim*2, kernel_size=2, stride=2)
        self.dec2 = TSMambaBlock(base_dim*2)

        self.up1 = nn.ConvTranspose3d(
            base_dim*2, base_dim, kernel_size=2, stride=2)
        self.dec1 = TSMambaBlock(base_dim)

        # Final Reconstruction Head
        self.final_up = nn.ConvTranspose3d(
            base_dim, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder path with skip connections
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(self.down1(e1))

        b = self.bottleneck(self.down2(e2))

        # Decoder path [9]
        d2 = self.dec2(self.up2(b) + e2)  # Skip connection [11]
        d1 = self.dec1(self.up1(d2) + e1)

        out = self.final_up(d1)
        return out, [e1, e2, b], [d1, d2]


class DualDomainLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.4):
        super().__init__()
        self.alpha = alpha  # Feature alignment weight
        self.beta = beta    # Data-space Huber reconstruction weight
        self.huber = nn.HuberLoss()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, input_vol, target_vol, enc_feats, dec_feats):
        # 1. Voxel-space reconstruction loss
        l_data = self.huber(input_vol, target_vol)

        # 2. Feature-space cosine distance
        l_feat = 0.0
        # Correctly matches e1 with d1 (base_dim) and e2 with d2 (base_dim*2)
        for f_e, f_d in zip(enc_feats[:2], dec_feats):
            sim = self.cosine(f_e, f_d).mean()
            l_feat += (1.0 - sim)

        return self.alpha * l_feat + self.beta * l_data

if __name__ == "__main__":
    #loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MambaUAD().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = DualDomainLoss(alpha=1, beta=0.4)
    epochs = 5

    calculate_psnr = PeakSignalNoiseRatio(data_range=5.0).to(device)

    loss_history = []
    vram_history = []
    psnr_history = []
    timestamps = []

    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        for idx, input_data in enumerate(train_loader):
            input_data = input_data.to(device)
            optimizer.zero_grad()
            output, encs, decs = model(input_data)
            loss = criterion(input_vol=output, target_vol=input_data, enc_feats=encs, dec_feats=decs)
            loss.backward()
            optimizer.step()
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            peak_vram_mb = peak_vram_bytes / (1024 ** 2)
            vram_history.append(peak_vram_mb)
            timestamps.append(time.time() - start_time)
            with torch.no_grad():
                psnr_val = calculate_psnr(output, input_data).item()
            psnr_history.append(psnr_val)
            loss_history.append(loss.item())
            print(peak_vram_mb)
            running_loss += loss.item()
            torch.cuda.reset_peak_memory_stats()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'mamba_weights.pth')

    plt.figure(1)
    plt.plot(timestamps, vram_history, color='blue', linewidth=2)
    plt.title('VRAM Usage Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('VRAM Allocated (MB)')
    plt.grid(True)
    plt.savefig('figures/mamba_vram_usage.svg', format='svg')
    plt.close()

    plt.figure(2)
    plt.plot(range(len(loss_history)), loss_history, color='blue', linewidth=2)
    plt.title('Training loss over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('figures/mamba_convergence.svg', format='svg')
    plt.close()

    plt.figure(3)
    plt.plot(range(len(loss_history)), psnr_history, color='green', linewidth=2)
    plt.title('PSNR over time')
    plt.xlabel('Batches (batch size = 1)')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig('figures/mamba_psnr.svg', format='svg')
    plt.close()