import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super(FeedForward, self).__init__()

        self.ffn = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, inputs: torch.Tensor):
        return self.ffn(inputs)
