import torch
import torch.nn as nn


class GatedSpatialConv3d(nn.Module):
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
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        L = D * H * W  # Total sequence length (e.g., 260k for 64^3)

        # Orientation 1: Forward Direction
        x_f = x.permute(0, 2, 3, 4, 1).view(B, L, C)
        y_f = self.mamba(x_f)
        # y_f: (B, D*H*W, C) -> view to (B, D, H, W, C) -> permute to (B, C, D, H, W)
        y_f = y_f.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Orientation 2: Reverse Direction (Flipped along the sequence dimension, which is index 1)
        x_r = x_f.flip(dims=[1])
        y_r = self.mamba(x_r).flip(dims=[1])
        y_r = y_r.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Orientation 3: Inter-slice Direction (Transpose D and W)
        x_s = x.permute(0, 4, 3, 2, 1).view(B, L, C)
        y_s = self.mamba(x_s)
        # y_s: (B, W*H*D, C) -> view back to transposed shape (B, W, H, D, C) -> permute back to original (B, C, D, H, W)
        y_s = y_s.view(B, W, H, D, C).permute(0, 4, 3, 2, 1)

        # ToM(z) = Mamba(zf) + Mamba(zr) + Mamba(zs)
        return y_f + y_r + y_s


class TSMambaBlock(nn.Module):
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


class MambaUAD(nn.Module):
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
        self.alpha = alpha  # Feature weight [14]
        self.beta = beta   # Data weight [14]
        self.huber = nn.HuberLoss()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, input_vol, target_vol, enc_feats, dec_feats):
        # 1. Data-space reconstruction (Huber Loss) [12, 13]
        l_data = self.huber(input_vol, target_vol)

        # 2. Feature-space reconstruction (Cosine Similarity) [13, 15]
        l_feat = 0
        for f_e, f_d in zip(enc_feats[:2], dec_feats[::-1]):
            # Align multi-scale features [12]
            sim = self.cosine(f_e, f_d).mean()
            l_feat += (1 - sim)

        return self.alpha * l_feat + self.beta * l_data
