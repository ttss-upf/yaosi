import json
import collections
import re

r = "[!%^,.?~@#$%……&*{}()\\\"/]"
train_input_file_path = "../data/processed_data/eng/train_src"
train_output_file_path = '../data/tokenized/eng/train_tgt.tok'

def get_words(data):
    words_box=[]

    for line in data:
        line = re.sub(r, '', line)
        words_box.extend(line.replace(' <TSP> ',' ').replace(' | ', ' ').replace('_', ' ').lower().split())

    return collections.Counter(words_box)


def main():

    with open(train_output_file_path,'r') as f:
        train_outputs = f.readlines()
    with open(train_input_file_path,'r') as f:
        train_inputs = f.readlines()
        
    output_vocab = get_words(train_outputs)
    input_vocab = get_words(train_inputs)
    input_vocab = [key for key, _ in input_vocab.most_common()]
    output_vocab = [key for key, _ in output_vocab.most_common()]
    all_vocab = input_vocab + output_vocab
    all_vocab = ['<pad>','<start>','<end>','<unk>','A0','A1','NE']+all_vocab

    print(len(all_vocab))

    word2id = {}

    for idx, line in enumerate(all_vocab):
        word2id[line] = idx
        
    vocab_file_path1 = '../data/vocab/vocab_word2id'
    vocab_file_path2 = '../data/vocab/vocab_id2word'

    with open(vocab_file_path1,'w+') as f:
        json.dump(word2id,f)

    with open(vocab_file_path2,'w+') as f:
        json.dump(output_vocab,f)

    return word2id

main()