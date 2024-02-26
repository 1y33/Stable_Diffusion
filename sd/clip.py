import torch
from torch import nn

from sd.attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self,n_vocab,n_embed,n_tokens):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab,n_embed)
        self.position_embedding =  nn.Parameter(torch.zeros(n_tokens,n_embed))

    def forward(self,tokens):
        # bs ,seq_l  -> bs,seq_len , dim

        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self,n_head,n_embed):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head,n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed,4*n_embed)
        self.linear_2 = nn.Linaer(n_embed*4,n_embed)

    def forward(self,x):
        residual = x

        # self attentions
        x = self.layer_norm_1(x)
        x = self.attention(x,causal_mask=True)

        x += residual

        # ff layer
        residual = x
        x = self.layer_norm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function

        x = self.linear_2(x)

        x+= residual

        return x







class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768,77)

        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens : torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # bs , seq_l -> bs , seq_l , dim
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output