import argparse
import time
import torch
from transformer import get_model
import torch.nn.functional as F
from tool import generate_square_subsequent_mask, get_batch
import pickle
import math
from tqdm import tqdm
import torch.nn as nn
from utils import config
from utils.build_vocab import load_vocab


def train_model(model, opt):
    model.train()
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    _mask = generate_square_subsequent_mask(opt.max_length)
    num_batches = opt.num_batches  # should be calculated automatically

    print("loading vocab...")
    # src_file = "data/processed_data/" + opt.lang + "/train_src"
    # tgt_file = "data/processed_data/" + opt.lang + "/train_tgt"
    # src = torch.tensor([[1,2,3],[2,3,4]])
    # tgt = torch.tensor([[999,1,33],[3,4,5]])
    # src = []
    # tgt = []

    # with open(src_file, "r") as f:
    #     for i in tqdm(f.readlines()):
    #         src.append(load_vocab(i))
            

    # with open(tgt_file, "r") as f:
    #     for i in tqdm(f.readlines()):
    #         tgt.append(load_vocab(i))
            
    # torch.save(src, "data/vocab/vocab_src")
    # torch.save(tgt, "data/vocab/vocab_tgt")
    src = torch.load("data/vocab/vocab_src")
    tgt = torch.load("data/vocab/vocab_tgt")
    print("-- vocab loaded --")
    print("training model...")
    for epoch in range(opt.epochs):
        total_loss = 0
        # if opt.floyd is False:
        #     print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
        #     ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        print("epoch {}:".format(epoch))
        for batch, i in tqdm(
                enumerate(range(0, opt.num_batches, opt.interval))):
            print("_mask.shape,{}".format(_mask.shape))

            data, target = get_batch(src, tgt, i, opt)
            seq_len = data.size(0)
            # if seq_len != opt.interval:  # only on last batch
            #     _mask = _mask[:seq_len, :seq_len]

            preds = model(data, target, _mask, _mask)
            opt.optimizer.zero_grad()
            loss = criterion(preds.view(-1, opt.vocab_size), tgt)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()

            total_loss += loss.item()
            if batch % opt.interval == 0 and batch > 0:
                # lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / opt.interval
                cur_loss = total_loss / opt.interval
                ppl = math.exp(cur_loss)
                print(
                    f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-lr', type=float, default=0.0001)
    # parser.add_argument('-floyd', action='store_true')
    parser.add_argument("-d_model", type=int, required=True)
    parser.add_argument("-n_layers", type=int, required=True)
    parser.add_argument("-heads", type=int, required=True)
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-vocab_size", type=int, required=True)
    parser.add_argument("-dropout", type=float, required=True)
    parser.add_argument("-interval", type=int, required=True)
    parser.add_argument("-max_length", type=int, required=True)
    parser.add_argument("-lang", required=True)
    parser.add_argument("-num_batches", type=int, required=True)
    parser.add_argument('-no_cuda', action='store_true', default=False)

    opt = parser.parse_args()
    # print(opt)
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        # assert torch.cuda.is_available()
        print("running on GPU...")
    else:
        print("running on CPU...")

    model = get_model(opt, opt.vocab_size,
                      opt.max_length)  # 输入是什么？？是长度还是序列？

    opt.optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     betas=(0.9, 0.98),
                                     eps=1e-9)

    train_model(model, opt)


#     if opt.floyd is False:
#         promptNextAction(model, opt, SRC, TRG)

# def yesno(response):
#     while True:
#         if response != 'y' and response != 'n':
#             response = input('command not recognised, enter y or n : ')
#         else:
#             return response

# def promptNextAction(model, opt, SRC, TRG):

#     saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0

#     if opt.load_weights is not None:
#         dst = opt.load_weights
#     if opt.checkpoint > 0:
#         dst = 'weights'

#     while True:
#         save = yesno(input('training complete, save results? [y/n] : '))
#         if save == 'y':
#             while True:
#                 if saved_once != 0:
#                     res = yesno("save to same folder? [y/n] : ")
#                     if res == 'y':
#                         break
#                 dst = input(
#                     'enter folder name to create for weights (no spaces) : ')
#                 if ' ' in dst or len(dst) < 1 or len(dst) > 30:
#                     dst = input(
#                         "name must not contain spaces and be between 1 and 30 characters length, enter again : "
#                     )
#                 else:
#                     try:
#                         os.mkdir(dst)
#                     except:
#                         res = yesno(
#                             input(dst +
#                                   " already exists, use anyway? [y/n] : "))
#                         if res == 'n':
#                             continue
#                     break

#             print("saving weights to " + dst + "/...")
#             torch.save(model.state_dict(), f'{dst}/model_weights')
#             if saved_once == 0:
#                 pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
#                 pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
#                 saved_once = 1

#             print("weights and field pickles saved to " + dst)

#         res = yesno(input("train for more epochs? [y/n] : "))
#         if res == 'y':
#             while True:
#                 epochs = input("type number of epochs to train for : ")
#                 try:
#                     epochs = int(epochs)
#                 except:
#                     print("input not a number")
#                     continue
#                 if epochs < 1:
#                     print("epochs must be at least 1")
#                     continue
#                 else:
#                     break
#             opt.epochs = epochs
#             train_model(model, opt)
#         else:
#             print("exiting program...")
#             break

#     # for asking about further training use while true loop, and return

if __name__ == "__main__":
    main()
