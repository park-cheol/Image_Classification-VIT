import torch
import torch.nn as nn

from models.Attention import Attention
from models.FFN import FeedForward
from models.modules import PreNorm


class Transformer(nn.Module):

    def __init__(self, input_dim: int, n_layers: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim=input_dim, function=Attention(input_dim=input_dim, heads=heads,
                                                          dim_head=dim_head, dropout=dropout)),
                PreNorm(dim=input_dim, function=FeedForward(dim=input_dim, hidden_dim=mlp_dim,
                                                            dropout=dropout))
            ]))

    def forward(self, x: torch.Tensor):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x

        return x
