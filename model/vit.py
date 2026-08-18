import torch
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from dataset import train_loader, val_loader
from torchmetrics.functional.image import peak_signal_noise_ratio
import matplotlib.pyplot as plt
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
        reconstructed_img, hidden_states = self.model(x) 
        return reconstructed_img, hidden_states

class VitDualDomainLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta  
        self.huber = nn.HuberLoss()
        self.cosine = nn.CosineSimilarity(dim=-1)

    def forward(self, input_vol, target_vol, hidden_states):
        #Voxel-space reconstruction loss
        l_data = self.huber(input_vol, target_vol)

        l_feat = torch.tensor(0.0, device=input_vol.device)
        if hidden_states is not None and len(hidden_states) > 1:
            # Compare earlier layers against the deepest layer
            ref_layer = hidden_states[-1]
            for state in hidden_states[:-1]:
                sim = self.cosine(state, ref_layer).mean()
                l_feat += (1.0 - sim)
            l_feat = l_feat / (len(hidden_states) - 1) 

        return self.alpha * l_feat + self.beta * l_data
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #need cuda
    model = VitAE().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = VitDualDomainLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
    epochs = 15

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
            input_data = input_data.squeeze(dim=0).to(device) if input_data.ndim > 5 else input_data.to(device)
            optimizer.zero_grad()
            output, hidden_state_tensor = model(input_data)
            loss = criterion(input_vol=output, target_vol=input_data, hidden_states=hidden_state_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                psnr_val = peak_signal_noise_ratio(output, input_data, data_range=5.0).item()
                running_loss += loss.item()
                running_psnr += psnr_val
                
                if idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f'Batch {idx} Loss: {loss.item():.4f} PSNR: {psnr_val:.2f} current lr: {current_lr}')

        torch.cuda.synchronize(device)
        epoch_max_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        vram_history.append(epoch_max_vram_mb)
        model.eval()
        val_running_loss = 0.0
        val_running_psnr = 0.0
        with torch.no_grad():
            for idx, input_data in enumerate(val_loader):
                input_data = input_data.squeeze(dim=0).to(device) if input_data.ndim > 5 else input_data.to(device)
                output, hidden_state_tensor = model(input_data)
                loss = criterion(input_vol=output, target_vol=input_data, hidden_states=hidden_state_tensor)
                psnr_val = peak_signal_noise_ratio(output, input_data, data_range=5.0).item()
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
        
        print(f"Epoch [{epoch+1}/{epochs}], peak vram: {epoch_max_vram_mb} avg loss: {avg_loss:.4f}, avg val loss: {avg_val_loss}, avg psnr: {avg_psnr}, avg_val_psnr: {avg_val_psnr}")

        scheduler.step(avg_val_psnr)
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save(model.state_dict(), 'weights/best_vit_weights.pth')

    torch.save(model.state_dict(), 'weights/end_vit_weights.pth')

    epoch_axis = [num+1 for num in range(15)]

    plt.figure(1)
    plt.plot(epoch_axis, loss_history, color='blue', linewidth=2, label='Training loss')
    plt.plot(epoch_axis, val_loss_history, color='orange', linestyle='--', linewidth=2, label='Validation loss')
    plt.title('Training loss vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/vit_convergence.svg', format='svg')
    plt.close()

    plt.figure(2)
    plt.plot(epoch_axis, psnr_history, color='green', linewidth=2, label='Training PSNR')
    plt.plot(epoch_axis, val_psnr_history, color='red', linestyle='--', linewidth=2, label='Validation PSNR')
    plt.title('PSNR vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/vit_psnr.svg', format='svg')
    plt.close()

    plt.figure(3)
    plt.plot(epoch_axis, vram_history, color='purple', linewidth=2)
    plt.title('VRAM vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('VRAM usage (MB)')
    plt.xticks(epoch_axis)
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/vit_vram.svg', format='svg')
    plt.close()