import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange # einops.rearrange랑 다른 점이 F.softmax <=> nn.Softmax 차이와 같은 것 같다

from models.Transformer import Transformer
from utils.pair import pair


class VIT(nn.Module):

    def __init__(self, image_size, patch_size, num_classes, dim, n_layers, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(VIT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches +1 , dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # classification 하기 위해서 cls 추가
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(input_dim=dim, n_layers=n_layers, heads=heads, dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=dropout)
        self.pool = pool
        self.identity = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes))

    def forward(self, inputs: torch.Tensor):
        """
        inputs = [Batch, 3, 224, 224]
        """
        x = self.patch_embedding(inputs)

        b, n, _ = x.size()

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) # batch_size 맞추기
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)] # cls token 빼고?
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.identity(x) # todo why?
        return self.mlp_head(x)


if __name__ == "__main__":
    t = torch.randn(1, 3, 224, 224).cuda()
    v = VIT(image_size=224,
            patch_size=16,
            num_classes=100,
            dim=1024,
            n_layers=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1).cuda()
    print(v(t).size())
