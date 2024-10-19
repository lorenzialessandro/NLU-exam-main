# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import * # Import everything from functions.py file
from utils import read_file
import argparse

# define parameters
lr = 0.0005
runs = 1
hid_size = 200    # size of hidden layer
emb_size = 300    # size of embedding layer
epochs = 200
clip = 5
patience = 3
device = 'cuda:0'

import wandb
import math
import random
wandb.login(key="b538d8603f23f0c22e0518a7fcef14eef2620e7d")

def main():
    use_dropout = False # Adding dropout layer
    parser = argparse.ArgumentParser(description="Language Modeling Task")
    parser.add_argument('architecture', choices=['RNN', 'LSTM'], help="Choose the model architecture (RNN or LSTM)")
    parser.add_argument('optimizer', choices=['SGD', 'AdamW'], help="Choose the optimizer (SGD or AdamW)")
    parser.add_argument('--dropout', action='store_true', help="Include dropout in the model")
    args = parser.parse_args()
    
    model = args.architecture
    optimizer = args.optimizer
    if args.dropout:
        use_dropout = True
    
    # Load the datasets
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")
    
    wandb.init(
        project="LM",
        config={
            "lr": lr,
            "runs": runs,
            "hid_size": hid_size,
            "emb_size": emb_size,
            "epochs": epochs,
            "clip": clip,
            "patience": patience,
            "device": device,
            "model": model,
            "optimizer": optimizer,
            "use_dropout": use_dropout,
            "loss": "CrossEntropyLoss",
            "metric": "Perplexity",
            "dataset": "PTB",
            "batch_size": 256,
        }

    )
    
    # Lunch the run(s) with the parameters
    run(train_raw, dev_raw, test_raw, lr=lr, runs=runs, epochs=epochs, clip=clip, patience=patience, device=device, hid_size=hid_size, emb_size=emb_size, model_type=model, optimizer_type=optimizer, use_dropout=use_dropout)

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    main()

