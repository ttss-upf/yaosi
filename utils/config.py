import os

root_dir = os.path.expanduser("~")

seperator = "<TSP>" # seperator, seems to be useless

# depends on the language
lang = "eng"
max_length = 128
# max_length = 10
vocab_size = 18751 # vocabulory size, depends on how many words are there
num_batches = 34352

# transformer model hyperparameter
d_model = 300 # embedding dimension
epoch = 3
dropout = 0.01
heads = 2
N = 2
batch_size = 16



