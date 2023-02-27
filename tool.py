import torch
from torchtext import data
# from torchtext.legacy import data
import numpy as np
from torch.autograd import Variable
from torch import Tensor
from typing import Tuple


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def get_batch(src, tgt, i, opt) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """

    seq_len = min(opt.batch_size, len(src) - 1 - i)
    # data = src[i:i + seq_len].view(config.max_length, 1)
    # target = tgt[i + 1:i + 1 + seq_len].view(config.max_length, 1)
    data = src[i:i + seq_len]
    target = tgt[i + 1:i + 1 + seq_len]
    return data, target
