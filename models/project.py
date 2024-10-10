import torch.nn.functional as F
import torch.nn as nn
import torch


class ProjectionNetwork(nn.Module):
    def __init__(self, dim=512):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, dim*2)
        self.fc2 = nn.Linear(dim*2, dim*2)
        self.fc3 = nn.Linear(dim*2, dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x + residual


# class GatedResidualBlock(nn.Module):
#     def __init__(self, dim):
#         super(GatedResidualBlock, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim*4)
#         self.fc3 = nn.Linear(dim*4, dim)
#         self.gate = nn.Linear(dim, dim)
#         self.activation = nn.GELU()
#
#     def forward(self, x):
#         residual = x
#         out = self.activation(self.fc1(x))
#         out = self.activation(self.fc2(out))
#         out = self.fc3(out)
#         gate = torch.sigmoid(self.gate(x))
#         out = gate * out + (1 - gate) * residual
#         return out
#
#
# class ProjectionNetwork(nn.Module):
#     def __init__(self, dim=512, num_blocks=8):
#         super(ProjectionNetwork, self).__init__()
#         self.input_proj = nn.Linear(dim, dim)
#         self.blocks = nn.ModuleList([GatedResidualBlock(dim) for _ in range(num_blocks)])
#         self.output_proj = nn.Linear(dim, dim)
#
#     def forward(self, x):
#         x = self.input_proj(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.output_proj(x)
#         return x