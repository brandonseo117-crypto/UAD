import torch
import torch.nn as nn
from einops import rearrange
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from dataset import train_loader, val_loader
from mamba_ssm import Mamba
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
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
        
        x_f = rearrange(x, 'b c d h w -> b (d h w) c')
        y_f = self.mamba_f(x_f)
        y_f = rearrange(y_f, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        x_r = torch.flip(x_f, dims=[1])
        y_r = self.mamba_r(x_r)
        y_r = torch.flip(y_r, dims=[1])
        y_r = rearrange(y_r, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        x_s_perm = rearrange(x, 'b c d h w -> b c w h d')
        x_s = rearrange(x_s_perm, 'b c w h d -> b (w h d) c')
        y_s = self.mamba_s(x_s)
        y_s = rearrange(y_s, 'b (w h d) c -> b c w h d', w=W, h=H, d=D)
        y_s = rearrange(y_s, 'b c w h d -> b c d h w')
        
        # Fusion
        return y_f + y_r + y_s


class TSMambaBlock(nn.Module): # good
    def __init__(self, dim):
        super().__init__()
        self.gsc = GatedSpatialConv3d(dim)
        self.tom = TriOrientatedMamba(d_model=dim)
        self.norm1 = nn.InstanceNorm1d(dim)
        self.norm2 = nn.InstanceNorm1d(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = self.gsc(x)

        B, C, D, H, W = x.shape
        res = x
        x = rearrange(x, 'b c d h w -> b c (d h w)')
        x = self.norm1(x)
        x = rearrange(x, 'b c (d h w) -> b c d h w', d=D, h=H, w=W)
        x = self.tom(x) + res
        
        res = x
        x = rearrange(x, 'b c d h w -> b c (d h w)')
        x = self.norm2(x)
        
        x = rearrange(x, 'b c (d h w) -> b (d h w) c')
        x = self.mlp(x)
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        return x + res


class MambaUAD(nn.Module): #Good
    def __init__(self, in_channels=1, base_dim=48):
        super().__init__()
        # Stem: 7x7x7 Depth-wise Conv, Stride 2 [10]
        self.stem = nn.Conv3d(in_channels, base_dim,kernel_size=7, stride=2, padding=3)

        # Encoder Layers [11]
        self.enc1 = TSMambaBlock(base_dim)
        self.down1 = nn.Conv3d(base_dim, base_dim*2, kernel_size=2, stride=2)
        self.enc2 = TSMambaBlock(base_dim*2)
        self.down2 = nn.Conv3d(base_dim*2, base_dim*4, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = TSMambaBlock(base_dim*4)

        # Gating mechanism
        self.gate1 = GatedSpatialConv3d(base_dim)
        self.gate2 = GatedSpatialConv3d(base_dim*2)

        # Decoder Layers (Symmetric) [9]
        self.up2 = nn.ConvTranspose3d(base_dim*4, base_dim*2, kernel_size=2, stride=2)
        self.dec2 = TSMambaBlock(base_dim*2)
        self.up1 = nn.ConvTranspose3d(base_dim*2, base_dim, kernel_size=2, stride=2)
        self.dec1 = TSMambaBlock(base_dim)

        # Final Reconstruction Head
        self.final_up = nn.ConvTranspose3d(base_dim, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder path with GATED skip connections
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        up_b = self.up2(b)
        gated_e2 = self.gate2(e2)
        d2 = self.dec2(up_b + gated_e2)
        up_d2 = self.up1(d2)
        gated_e1 = self.gate1(e1)
        d1 = self.dec1(up_d2 + gated_e1)

        out = self.final_up(d1)
        return out, [e1, e2], [d1, d2]


class DualDomainLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.huber = nn.HuberLoss()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, input_vol, target_vol, enc_feats, dec_feats):
        l_data = self.huber(input_vol, target_vol)
        l_feat = 0.0
        for f_e, f_d in zip(enc_feats, dec_feats):
            dist = 1.0 - self.cosine(f_e, f_d)
            l_feat += dist.mean()
        
        return self.alpha * l_feat + self.beta * l_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #need cuda
    model = MambaUAD().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = DualDomainLoss(alpha=1, beta=0.4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    epochs = 20

    vram_history = []
    loss_history = []
    val_loss_history = []
    val_psnr_history = []
    val_ssim_history = []

    best_val_psnr = -float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        torch.cuda.reset_peak_memory_stats(device)
        for idx, input_data in enumerate(train_loader):
            input_data = input_data.squeeze(dim=0).to(device) if input_data.ndim > 5 else input_data.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, encs, decs = model(input_data)
            loss = criterion(input_vol=output, target_vol=input_data, enc_feats=encs, dec_feats=decs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if idx % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f'Batch {idx} Loss: {loss.item():.4f} current lr: {current_lr}')

        torch.cuda.synchronize(device)
        epoch_max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        vram_history.append(epoch_max_vram_mb)
        model.eval()
        val_running_loss = 0.0
        val_running_psnr = 0.0
        val_running_ssim = 0.0
        with torch.no_grad():
            for idx, input_data in enumerate(val_loader):
                input_data = input_data.squeeze(dim=0).to(device) if input_data.ndim > 5 else input_data.to(device)
                output, encs, decs = model(input_data)
                loss = criterion(input_vol=output, target_vol=input_data, enc_feats=encs, dec_feats=decs)
                psnr_val = peak_signal_noise_ratio(output, input_data, data_range=(input_data.max() - input_data.min()).item()).item()
                ssim_val = structural_similarity_index_measure(output, input_data, data_range=(input_data.max() - input_data.min()).item()).item()
                val_running_loss += loss.item()
                val_running_psnr += psnr_val
                val_running_ssim += ssim_val

        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        avg_val_psnr = val_running_psnr / len(val_loader)
        val_psnr_history.append(avg_val_psnr)
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        avg_val_ssim = val_running_ssim / len(val_loader)
        val_ssim_history.append(avg_val_ssim)
        
        print(f"Epoch [{epoch+1}/{epochs}], peak vram: {epoch_max_vram_mb:.1f} avg loss: {avg_loss:.4f}, avg val loss: {avg_val_loss:.4f}, avg val ssim: {avg_val_ssim:.4f}, avg_val_psnr: {avg_val_psnr:.2f}")
        scheduler.step(avg_val_psnr)
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(model.state_dict(), 'weights/best_mamba_weights.pth')

    torch.save(model.state_dict(), 'weights/end_mamba_weights.pth')

    epoch_axis = [num+1 for num in range(epochs)]

    plt.figure(1)
    plt.plot(epoch_axis, loss_history, color='blue', linewidth=2, label='Training loss')
    plt.plot(epoch_axis, val_loss_history, color='orange', linestyle='--', linewidth=2, label='Validation loss')
    plt.title('Training loss vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/mamba_convergence.svg', format='svg')
    plt.close()

    plt.figure(2)
    plt.plot(epoch_axis, val_psnr_history, color='red', linestyle='--', linewidth=2)
    plt.title('Validation PSNR vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.savefig('figures/mamba_psnr.svg', format='svg')
    plt.close()

    plt.figure(3)
    plt.plot(epoch_axis, vram_history, color='purple', linewidth=2)
    plt.title('VRAM vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('VRAM usage (MB)')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.savefig('figures/mamba_vram.svg', format='svg')
    plt.close()

    plt.figure(4)
    plt.plot(epoch_axis, val_ssim_history, color='red', linewidth=2)
    plt.title('Validation SSIM vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.savefig('figures/mamba_ssim.svg', format='svg')
    plt.close()