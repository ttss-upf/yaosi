import json
import collections
import re
import torch
from utils import config
import numpy as np

train_input_file_path = '../data/processed_data/' + config.lang + "/train_src"
train_output_file_path = '../data/processed_data/' + config.lang + "/train_tgt"

r = "[_!%^,.?~@#$%……&*<>{}()/]"


def get_words(data):
    words_box = []

    for line in data:
        line = re.sub(r, '', line)
        words_box.extend(
            line.replace(' <TSP> ', ' ').replace(' | ', ' ').lower().split())

    return collections.Counter(words_box)


def create_vocab():
    print("building vocab ...")
    with open(train_input_file_path, 'r') as f:
        train_inputs = f.readlines()

    with open(train_output_file_path, 'r') as f:
        train_outputs = f.readlines()

    input_vocab = get_words(train_inputs)
    # input_vocab = [key for key, _ in input_vocab.most_common()]

    output_vocab = get_words(train_outputs)
    # output_vocab = [key for key, _ in output_vocab.most_common()]

    all_vocab = input_vocab + output_vocab
    all_vocab = [key for key, _ in all_vocab.most_common()]
    all_vocab = ['<pad>', '<start>', '<end>', '<unk>'] + all_vocab

    print("Total token num: {}".format(len(all_vocab)))

    word2id = {}

    for idx, line in enumerate(all_vocab):
        word2id[line] = idx
        # id2word[idx] = line

    vocab_file_path1 = '../data/vocab/vocab_word2id'
    vocab_file_path2 = '../data/vocab/vocab_id2word'

    with open(vocab_file_path1, 'w+') as f:
        json.dump(word2id, f)

    with open(vocab_file_path2, 'w+') as f:
        json.dump(all_vocab, f)
    print("vocab built")
    return word2id


def load_vocab(text):
    data = {}
    arr = torch.zeros(config.max_length, dtype=int)
    with open("./data/vocab/vocab_word2id") as f:
        data = f.readline()
    obj = json.loads(data)
    text = re.sub(r, '', text)
    texts = text.replace(' <TSP> ', ' ').replace(' | ', ' ').lower().split()
    for k, v in enumerate(texts):
        arr[k] = obj[v]
    return arr
    # return torch.tensor(arr)


# if __name__ == "__main__":
#     result = load_vocab("Alan_Shepard | awards | Distinguished_Service_Medal_(United_States_Navy)")

#     print(result)
