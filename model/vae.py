import torch
from torch.optim import Adam
from monai.networks.nets import VarAutoEncoder

class VAEModel(torch.nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 128), latent_dim=256):
        super(VAEModel, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.vae = VarAutoEncoder(
            spatial_dims=3,
            in_channels=input_shape[0],
            out_channels=input_shape[0],
            latent_size=latent_dim,
            channels=(64, 128, 256),
            strides = (2,2),
            num_res_units=2
        )

    def forward(self, x):
        return self.vae(x)

    def compute_loss(self, x):
        recon_x, mu, logvar = self.forward(x)
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kld_loss)

    def anomaly_mapping(self, x):
        recon_x, _, _ = self.forward(x)
        return torch.abs(x - recon_x), recon_x 

#loop
model = VAEModel(input_shape=(1, 128, 128, 128), latent_dim=256)
model.train()
optimizer = Adam(model.parameters(), lr=1e-3)
epochs = 5

#pretend we have a dataloader called 'train_loader'
for epoch in range(epochs):
    for batch in train_loader:
        loss = model.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")