from data import test_loader
from vae import VAEModel
from vit import ViTVAE
from architecture import GatedSpatialConv3d, TSMambaBlock, TriOrientatedMamba, MambaUAD
import torch
from torchmetrics.image import PeakSignalNoiseRatio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mamba_model = MambaUAD()
mamba_state_dict = torch.load('mamba_weights.pth', weights_only=True)
mamba_model.load_state_dict(mamba_state_dict)

vae_model = VAEModel().to(device)
vae_state_dict = torch.load('vae_weights.pth', weights_only=True)
vae_model.load_state_dict(vae_state_dict)

vitvae_model = ViTVAE().to(device)
vitvae_state_dict = torch.load('vit_weights.pth', weights_only=True)
vitvae_model.load_state_dict(vitvae_state_dict)

# VAEs
vae_loss_vals = []
vitvae_loss_vals = []
for i in range(2):
    running_loss = 0.0
    for idx, input_data in enumerate(test_loader):
        input_data = input_data.to(device)
        model = vitvae_model if len(vae_loss_vals) == len(test_loader.dataset) else vae_model
        loss = model.compute_loss(input_data)

        running_loss += loss.item()
        loss_vals = vitvae_loss_vals if len(vae_loss_vals) == len(test_loader.dataset) else vae_loss_vals
        loss_vals.append(loss.item())

    avg_loss = running_loss / len(test_loader)
    print(f"Avg: {avg_loss}:.4f")

# Mamba
mamba_loss_vals = []
for idx, input_data in enumerate(test_loader):
    input_data = input_data.to(device)
    loss = mamba_model.compute_loss(input_data)
      
    running_loss += loss.item()
    mamba_loss_vals.append(loss.item())

avg_loss = running_loss / len(test_loader)
print(f"Avg: {avg_loss}:.4f")