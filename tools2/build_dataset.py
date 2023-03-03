import re
import torch
import json
import numpy as np
import collections
train_src = []
train_tgt = []
lines = []
r = "[!%^,.?~@#$%……&*{}()\\\"/]"
train_input_file_path = "../data/processed_data/eng/train_src"
train_output_file_path = '../data/tokenized/eng/train_tgt.tok'
# word2id = json.load("../data/vocab/vocab_word2id")
print("writing...")
with open("../data/vocab/vocab_word2id") as f:
    data = f.readline()
obj = json.loads(data)
def load_vocab(text):
    data = {}
    arr = torch.zeros(128, dtype=int)

    text = re.sub(r, '', text)
    texts = text.replace(' <TSP> ', ' ').replace("_"," ").replace(' | ', ' ').lower().split()
    for k, v in enumerate(texts):
        arr[k] = obj[v]
    return arr
with open(train_input_file_path, "r") as f:
    lines = f.readlines()
for line in lines:
    train_src.append(load_vocab(line))
with open(train_output_file_path, "r") as f:
    lines = f.readlines()
for line in lines:
    train_tgt.append(load_vocab(line))
    
    
print("saving...")
train_src = torch.stack(train_src)
train_tgt = torch.stack(train_tgt)


torch.save(train_src, "../data/dataset/train_src")
torch.save(train_tgt, "../data/dataset/train_tgt")




