import copy
import torch.nn as nn
import torch.nn.functional as F
from TransformerLayers import TransformerDecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=memory_padding_mask)
        output = self.linear(output)
        return output