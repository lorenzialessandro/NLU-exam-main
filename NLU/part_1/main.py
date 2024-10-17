# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import * # Import everything from functions.py file
from utils import load_data
import argparse

# define parameters
bert_model = 'bert-base-uncased'
lr = 0.0001
runs = 5
n_epochs = 100
hid_size = 200    # size of hidden layer
emb_size = 300    # size of embedding layer
clip = 5
patience = 5
device = 'cuda:0'


def main():
    bidirectionality = False # Adding bidirectionality
    dropout_layer = False    # Adding dropout layer
    parser = argparse.ArgumentParser(description="Intent and Slot Filling Task")
    parser.add_argument('--bid', action='store_true', help='Optional flag to add bidirectionality')
    parser.add_argument('--drop', action='store_true', help='Optional flag to add dropout layer')
    args = parser.parse_args()
    
    if args.bid:
        bidirectionality = True
    if args.drop:
        dropout_layer = True
    
    # Load the datasets
    tmp_train_raw = load_data(os.path.join('dataset','train.json'))
    test_raw = load_data(os.path.join('dataset','test.json'))
    
    # Lunch the run(s) with the parameters
    run(tmp_train_raw, test_raw, bert_model=bert_model, lr=lr, runs=runs, n_epochs=n_epochs, clip=clip, patience=patience, device=device, hid_size=hid_size, emb_size=emb_size, bidirectionality=bidirectionality, dropout_layer=dropout_layer)
    

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    main()
    
