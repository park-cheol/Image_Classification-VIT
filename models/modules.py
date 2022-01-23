import torch
import torch.nn as nn


class PreNorm(nn.Module):

    def __init__(self, dim: int, function: nn.Module):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.function = function

    def forward(self, inputs: torch.Tensor, **kwargs):
        return self.function(self.norm(inputs), **kwargs)
