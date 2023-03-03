import torch
import torch.nn
import torchtext
from torchtext.data import get_tokenizer

# tokenizer = get_tokenizer("basic_english")
# tokens = tokenizer("Aarhus_Airport | cityServed | Aarhus, Denmark")

# train_src = torch.load("data/dataset/train_src")
train_src = torch.load("data/dataset/train_src")
print(train_src[:2])
# print(tokens)

