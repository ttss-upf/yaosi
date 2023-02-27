import os

root_dir = os.path.expanduser("~")

seperator = "<TSP>" # seperator, seems to be useless

# depends on the language
lang = "eng"
max_length = 128

# transformer model parameter
d_model = 300 # embedding dimension
vocab = 18751 # vocabulory size, depends on how many words are there
epoch = 3


