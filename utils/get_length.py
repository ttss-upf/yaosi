import sys
import config
import numpy
from tqdm import tqdm

filename = "../data/processed_data/" + config.lang + "/train_tgt"
lens = []
print("counting...")
with open(filename, 'r') as f:
    for ll in tqdm(f):
        lens.append(len(ll.strip().split(' ')))
print(filename, ' max ', numpy.max(lens), ' min ', numpy.min(lens), ' average ', numpy.mean(lens))