import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#N Dimensional - Fully connected - Score Net
class NDFC(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, sigmas=None):
        super().__init__()
        self.register_buffer('sigmas', sigmas)
        if self.sigmas is not None:
            current_dim = in_dim + 1
            self.conditioning_branch = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            current_dim = in_dim
            self.conditioning_branch = None

        layers = nn.ModuleList()
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            current_dim = dim

        self.layers = layers
        self.head = nn.Linear(current_dim, out_dim)

    def forward(self, x, t):
        if self.sigmas is not None:
            sigma_embed = self.conditioning_branch(self.sigmas[t][:,None])
            z = torch.cat((x, sigma_embed), dim=1)
        else:
            z = x
        for layer in self.layers:
            z = layer(z)
            z = F.relu(z)

        return self.head(z)

def get_medium_NDFC(in_dim, out_dim, sigmas=None):
    return NDFC(in_dim, [128, 256], out_dim, sigmas=sigmas)
def get_large_NDFC(in_dim, out_dim, sigmas=None):
    return NDFC(in_dim, [128, 256, 512], out_dim, sigmas=sigmas)
