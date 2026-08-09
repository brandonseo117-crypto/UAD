import torch
from monai.networks.nets import VariationalAutoEncoder

class VAEModel(torch.nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 128), latent_dim=256):
        super(VAEModel, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.vae = VariationalAutoEncoder(
            spatial_dims=3,
            in_channels=input_shape[0],
            out_channels=input_shape[0],
            latent_size=latent_dim,
            hidden_sizes=[32, 64, 128],
            num_res_units=2,
        )

    def forward(self, x):
        return self.vae(x)