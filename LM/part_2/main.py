# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import * # Import everything from functions.py file
from utils import read_file
import argparse

# define parameters
lr = 4
runs = 1
hid_size = 300    # size of hidden layer
emb_size = 300    # size of embedding layer
epochs = 200
clip = 5
patience = 5
device = 'cuda:0'

def main():
    var_dropout = False # Use variational dropout
    weight_tying = False # Use weight tying
    parser = argparse.ArgumentParser(description="Language Modeling Task")
    parser.add_argument('optimizer', choices=['SGD', 'NTAvSGD'], help="Choose the optimizer (SGD or NTAvSGD)")
    parser.add_argument('--weight_tying', action='store_true', help="Use weight tying in the model")
    parser.add_argument('--var_dropout', action='store_true', help="Use variational dropout in the model")
    args = parser.parse_args()
    
    optimizer = args.optimizer
    if args.weight_tying:
        weight_tying = True
    if args.var_dropout:
        var_dropout = True
    
    # Load the datasets
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")
    
    # Lunch the run(s) with the parameters
    run(train_raw, dev_raw, test_raw, lr=lr, runs=runs, epochs=epochs, clip=clip, patience=patience, device=device, hid_size=hid_size, emb_size=emb_size, optimizer_type=optimizer, weight_tying=weight_tying, var_dropout=var_dropout)

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    main()

