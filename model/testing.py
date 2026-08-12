from vae import VAEModel
from vit import ViTAutoEncoder
from architecture import GatedSpatialConv3d, TSMambaBlock, TriOrientatedMamba, MambaUAD
model.eval()

# VAEs
for idx, input in enumerate(test_loader):
        input = input.to(device)
        loss = model.compute_loss(input)

        running_loss += loss.item()
        loss=0

    avg_loss = running_loss / len(test_loader)
    print(f"Epoch [{epoch+1}/{epochs}], avg loss: {avg_loss:.4f}")