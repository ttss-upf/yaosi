import os
import time
import torch
from tqdm import tqdm
import torch.nn as nn
from tools2.build_tensor import builder
from torch.utils.data import DataLoader
from src.model.Transformer import Transformer


def _train_transformer(args):
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Running on {device}...")
    # embedding = nn.Embedding(args.vocab_size, args.emb_dim).to(device)

    src = torch.load("data/dataset/train_src").to(device)
    tgt = torch.load("data/dataset/train_tgt").to(device)
    # train_loader = DataLoader(src, batch_size=args.batch_size)
    train_loader = DataLoader(src, args.batch_size)

    model = Transformer(
        args.vocab_size,
        args.emb_dim,
        args.num_heads,
        args.enc_layers,
        args.max_len,
        dropout=args.dropout,
    ).to(device)
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # set the model to train mode
    model.train()
    for epoch in range(args.epochs):
        min_loss = float("inf")
        # loop over the batches in the dataset
        for i, inputs in enumerate(tqdm(train_loader)):
            targets = tgt[i * args.batch_size : (i + 1) * args.batch_size]
            ys = targets.contiguous().view(-1)
            # zero out the gradients

            # forward pass through the model
            outputs = model(inputs, targets)
            optimizer.zero_grad()
            # inputs = embedding(inputs)

            # print("output.value")
            # print(outputs)
            # print("outputs shape")
            # print(outputs.shape)
            # print("inputs shape")
            # print(inputs.shape)

            # target_tensor = builder(targets).to(device)
            # print(target_tensor.sum(dim=-1))
            # target_tensor = torch.zeros(1,128,12360).to(device)
            # target_tensor[:,:,9999] = 1.
            # print(target_tensor.shape)
            # print("max argmax")
            # print(torch.argmax(outputs, dim=-1))
            # print(torch.sum(torch.argmax(outputs, dim=-1),dim=0))
            # print(torch.argmax(outputs, dim=-1).shape)
            # print(torch.sum(outputs[0],dim=1))
            # print(target_tensor[0])
            # # compute the loss
            # print("output.value")
            # print(outputs)
            # print("target.tensor")
            # print(target_tensor)
            # print(outputs.sum(dim=-1))

            # print("output.shape{}".format(outputs.permute(0,2,1).shape))
            # print("targets.shape{}".format(targets.shape))
            # loss = criterion(outputs.permute(0,2,1), targets).mean(dim=1).mean(dim=0)
            # print(outputs.view(-1, outputs.size(-1)).shape)
            # loss = criterion(outputs.view(-1, outputs.size(-1)), targets.squeeze(0))
            loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), ys)
            
            # compute the gradients
            loss.backward()

            # clip the gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update the parameters
            optimizer.step()

            # print the loss every 100 batches
            if i % args.eval_steps == 0 and i != 0:
                print()
                print(
                    f"Epoch {epoch + 1} | Trained Batch {i} | Train Loss: {format(loss.item(), '.5f')} |"
                )
            if min_loss > loss:
                torch.save(model.encoder.state_dict(), "models/_encoder")
                torch.save(model.decoder.state_dict(), "models/_decoder")
                min_loss = loss
                # print("model saved...")


def _translate():
    print("translating...")
