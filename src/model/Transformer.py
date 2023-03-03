import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.EmbeddingLayers import TransformerEmbedding
from src.layers.TransformerLayers import TransformerEncoderLayer
from src.layers.TransformerLayers import TransformerDecoderLayer
class Transformer(nn.Module):
    def __init__(self, num_tokens, d_model, num_heads, num_layers, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = TransformerEmbedding(num_tokens, d_model, max_len, dropout=dropout)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, num_tokens)


    def forward(self, src, tgt):
        src_mask = self._generate_square_subsequent_mask(src.shape[0])
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(torch.int) & self._generate_tgt_mask(tgt.shape[0]).to(torch.int)
        tgt_mask = tgt_mask.to(torch.float)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        for enc_layer in self.encoder:
            src = enc_layer(src, src_mask)
        for dec_layer in self.decoder:
            tgt = dec_layer(tgt, src, tgt_mask, src_mask)
        out = self.out(tgt)
        out = F.softmax(out, dim=-1)
        # out = tgt
        return out

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda() if torch.cuda.is_available() else mask

    def _generate_tgt_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        return mask.cuda() if torch.cuda.is_available() else mask