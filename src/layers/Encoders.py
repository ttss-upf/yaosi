import torch.nn as nn
from TransformerLayers import TransformerEncoderLayer
import copy

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output