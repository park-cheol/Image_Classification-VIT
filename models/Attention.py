import torch
import torch.nn as nn

from einops import rearrange


class Attention(nn.Module):

    def __init__(self, input_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        hidden_dim = dim_head * heads # [512]
        proj = not (heads == 1 and dim_head == input_dim) # input_dim과 dim_head가 다르면 맞춤

        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(input_dim, hidden_dim * 3, bias=False)

        self.proj = nn.Sequential(nn.Linear(hidden_dim, input_dim),
                                  nn.Dropout(dropout)) if proj else nn.Identity()

    def forward(self, inputs: torch.Tensor):
        # Query Key Value 생성
        qkv = self.qkv(inputs).chunk(3, dim=-1)
        q, k, v = map(lambda  t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) # todo Print

        energy = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(energy)

        outputs = torch.matmul(attn, v)
        outputs = rearrange(outputs, 'b h n d -> b n (h d)')
        return outputs




