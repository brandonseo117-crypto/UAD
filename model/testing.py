from dataset import test_loader
from vae import VAEModel, ELBOvae
from vit import VitAE, VitDualDomainLoss
from architecture import MambaUAD, DualDomainLoss
import torch
from torchmetrics.image import PeakSignalNoiseRatio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mamba_model = MambaUAD()
mamba_state_dict = torch.load('mamba_weights.pth', weights_only=True)
mamba_model.load_state_dict(mamba_state_dict)
mamba_model.eval()

vae_model = VAEModel().to(device)
vae_state_dict = torch.load('vae_weights.pth', weights_only=True)
vae_model.load_state_dict(vae_state_dict)
vae_model.eval()

vitae_model = VitAE().to(device)
vitae_state_dict = torch.load('vit_weights.pth', weights_only=True)
vitae_model.load_state_dict(vitae_state_dict)
vitae_model.eval()


calculate_psnr = PeakSignalNoiseRatio(data_range=5.0).to(device)

#VAE
vae_loss_vals = []
psnr_vals = []
running_loss = 0.0
criterion = ELBOvae()
with torch.no_grad():
    for idx, (input_data, label, vis_date, pt_id) in enumerate(test_loader):
        input_data = input_data.to(device)
        output, mu, logvar = vae_model(input_data)
        loss = criterion(input_data, output, mu, logvar)
        running_loss += loss.item()
        vae_loss_vals.append(loss.item())
        psnr_vals.append(calculate_psnr(output, input_data).item())

avg_loss = running_loss / len(test_loader)
print(f"Avg: {avg_loss}:.4f")


# Mamba
mamba_loss_vals = []
criterion = DualDomainLoss()
running_loss = 0.0
with torch.no_grad():
    for idx, (input_data, label, vis_date, pt_id) in enumerate(test_loader):
        input_data = input_data.to(device)
        output, encs, decs = mamba_model(input_data)
        loss = criterion(input_vol=output, target_vol=input_data, enc_feats=encs, dec_feats=decs)
        running_loss += loss.item()
        mamba_loss_vals.append(loss.item())
        
        psnr_vals.append(calculate_psnr(output, input_data).item())

avg_loss = running_loss / len(test_loader)
print(f"Avg: {avg_loss}:.4f")

# VitAE

vit_loss_vals = []
criterion = VitDualDomainLoss()
running_loss = 0.0
with torch.no_grad():
    for idx, (input_data, label, vis_date, pt_id) in enumerate(test_loader):
        input_data = input_data.to(device)
        output, hidden_state = vitae_model(input_data)
        loss = criterion(input_vol=output, target_vol=input_data, hidden_states=hidden_state)
        running_loss += loss.item()
        vit_loss_vals.append(loss.item())
        psnr_vals.append(calculate_psnr(output, input_data).item())

avg_loss = running_loss / len(test_loader)
print(f"Avg: {avg_loss}:.4f")