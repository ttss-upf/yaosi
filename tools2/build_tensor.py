import torch
import torch.nn
import torchtext
from tqdm import tqdm
from torchtext.data import get_tokenizer

# train_tgt = torch.load("../data/dataset/train_tgt")

def builder(tgt):
    dim0, dim1 = tgt.shape
    result = torch.zeros(len(tgt), dim1, 12360, dtype=float)
    for batch in range(dim0):
        for row in range(dim1):
                k = tgt[batch][row]
                result[batch][row][k] = 1.
                # print(result[batch][row][k-1:k+3])
                    
    
    return result
    
