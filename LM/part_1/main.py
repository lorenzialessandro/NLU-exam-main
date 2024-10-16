# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import sys

# define parameters
device = 'cuda:0'
hid_size = 200    # size of hidden layer
emb_size = 300    # size of embedding layer
n_epochs = 100
patience = 3

def main():
    
    # Preprocess and load data
    train_loader, dev_loader, test_loader, lang = preprocess_and_load_data()
    vocab_len = len(lang.word2id)

    if len(sys.argv) < 3:
        print("Usage: python3 main.py <model> <optimizer>")
        return

    model = sys.argv[1]
    optimizer = sys.argv[2]
    lr = 0.01 # default

    # RNN with SGD
    # LSTM with SGD
    # LM_LSTM_dropout with SGD
    # LM_LSTM_dropout with AdamW

    if model == "RNN":
        if optimizer == "SGD":
            lr = 1.5
            model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            print("RNN can be used only with SGD")
            return
    elif model == "LSTM":
        if optimizer == "SGD":
            lr = 1.5
            model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            print("LSTM can be used only with SGD")
            return
    elif model == "LSTM_dropout":
        if optimizer == "SGD":
            lr = 1.3
            model = LM_LSTM_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            lr = 0.001
            model = LM_LSTM_dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        else:
            print("LSTM_dropout can be used only with SGD and AdamW")
            return 
    else: 
        print("Usage: python3 main.py <model> <optimizer>")
        return



    
    model.apply(init_weights)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')   
    
    # Train and evaluate the model
    result = train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, device, n_epochs, patience, lr)
    
    print(result)
    
    # path = 'model_bin/LSTM_dropout(AdamW).pt'
    # torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
        


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    main()

