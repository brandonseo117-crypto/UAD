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
