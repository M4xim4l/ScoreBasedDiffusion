import torch
import torch.nn as nn
import torch.nn.functional as F

#N Dimensional - Fully connected - Score Net
class NDFC(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, time_aware=False):
        super().__init__()
        self.time_aware = time_aware
        if time_aware:
            current_dim = in_dim + 1
        else:
            current_dim = in_dim
        layers = nn.ModuleList()
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            current_dim = dim

        self.layers = layers
        self.head = nn.Linear(current_dim, out_dim)

    def forward(self, x, t):
        if self.time_aware:
            z = torch.cat((x,t[:,None]), dim=1)
        else:
            z = x
        for layer in self.layers:
            z = layer(z)
            z = F.relu(z)

        return self.head(z)

def get_medium_non_timestep_aware(in_dim, out_dim):
    return NDFC(in_dim, [128, 256], out_dim)

def get_small_timestep_aware(in_dim, out_dim):
    return NDFC(in_dim, [128], out_dim, time_aware=True)

def get_medium_timestep_aware(in_dim, out_dim):
    return NDFC(in_dim, [128, 256], out_dim, time_aware=True)

def get_large_timestep_aware(in_dim, out_dim):
    return NDFC(in_dim, [128, 256, 256], out_dim, time_aware=True)
