import torch
import torch.nn as nn
from einops import rearrange
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from dataset import train_loader, val_loader
from mamba_ssm import Mamba
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import matplotlib.pyplot as plt

class GatedSpatialConvolution(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(nn.BatchNorm3d(in_channels), nn.SiLU(), nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1))
        self.conv1x1 = nn.Sequential(nn.BatchNorm3d(in_channels), nn.SiLU(), nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        self.fusion = nn.Sequential(nn.BatchNorm3d(in_channels), nn.SiLU(), nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.fusion(self.conv3x3(x) * self.conv1x1(x))

class ToM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mamba1 = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.mamba2 = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.mamba3 = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        B, C, D, H, W = x.shape
        z_f = rearrange(x, 'b c d h w -> b (d h w) c')
        out_f = self.mamba1(z_f)
        out_f = rearrange(out_f, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        
        z_r = rearrange(torch.flip(x, dims=[2,3,4]).contiguous(), 'b c d h w -> b (d h w) c')
        out_r = self.mamba2(z_r)
        out_r = rearrange(out_r, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        out_r = torch.flip(out_r, dims=[2, 3, 4])

        z_s = rearrange(x, 'b c d h w -> b (w h d) c')
        out_s = self.mamba3(z_s)
        out_s = rearrange(out_s, 'b (w h d) c -> b c d h w', w=W, h=H, d=D)

        return out_f + out_s + out_r

class MambaBlock(nn.Module):
    def __init__(self, channels=192):
        super().__init__()
        self.gsc = GatedSpatialConvolution(channels)
        self.ln1 = nn.LayerNorm(channels)
        self.tri = ToM(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels*2),
            nn.GELU(),
            nn.Linear(channels*2, channels)
            )
        
    def forward(self, x):
        gsc_out = self.gsc(x)
        ln1_in = rearrange(gsc_out, 'b c d h w -> b d h w c')
        ln1_out = self.ln1(ln1_in)
        ln1_out = rearrange(ln1_out, 'b d h w c -> b c d h w')
        tom_out = self.tri(ln1_out) + gsc_out
        ln2_in = rearrange(tom_out, 'b c d h w -> b d h w c')
        ln2_out = self.ln2(ln2_in)
        mlp_out = self.mlp(ln2_out)

        return rearrange(mlp_out, 'b d h w c -> b c d h w') + tom_out

class MambaModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        #encoder part
        self.stem = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.mambablock1 = MambaBlock(32)
        self.downsample = nn.Conv3d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.mambablock2 = MambaBlock(128)

        #decoder part
        self.mambablock3 = MambaBlock(128)
        self.upsample = nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.mambablock4 = MambaBlock(32)
        self.upsample2 = nn.ConvTranspose3d(in_channels=32, out_channels=out_channels, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.mambablock1(x)
        x = self.downsample(x)
        x = self.mambablock2(x)

        # Decoder
        x = self.mambablock3(x)
        x = self.upsample(x)
        x = self.mambablock4(x)
        out = self.upsample2(x)
        
        return out



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