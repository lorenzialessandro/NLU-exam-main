# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

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
    train_loader, dev_loader, test_loader, lang, train_dataset = preprocess_and_load_data()
    vocab_len = len(lang.word2id)
    total_samples = len(train_dataset)

    if len(sys.argv) < 3:
        print("Usage: python3 main.py <model> <optimizer>")
        return

    model = sys.argv[1]
    optimizer = sys.argv[2]

    # LSTM_weight_tying with SGD
    # LSTM_VariationalDropout with SGD
    # LSTM_VariationalDropout with NTAvSGD

    lr = 1.1         # learning rate
    emb_size = hid_size # for weight tying

    if model == "LSTM_weight_tying":
        if optimizer == "SGD":
            lr = 2
            model = LM_LSTM_VariationalDropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        else:
            print("LSTM_weight_tying can be used only with SGD")
            return
    elif model == "LSTM_VariationalDropout":
        if optimizer == "SGD":
            lr = 1.5
            model = LM_LSTM_VariationalDropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer == "NTAvSGD":
            lr = 1.3
            model = LM_LSTM_VariationalDropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
            optimizer = NTAvSGD(model.parameters(), lr=lr, total_samples=total_samples, batch_size=256)
        else:
            print("LSTM_VariationalDropout can be used with SGD or NTAvSGD")
            return
    else: 
        print("Usage: python3 main.py <model> <optimizer>")
        return
    
     
    model.apply(init_weights)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')   

    
    # Train and evaluate the model
    result = train_and_evaluate(train_loader, dev_loader, test_loader, optimizer, criterion_train, criterion_eval, model, device, n_epochs, patience)
    
    print(result)
    
    path = 'model_bin/LSTM_VariationalDropoutNTAvSGD.pt'
    torch.save(model.state_dict(), path)

    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
        

if __name__ == "__main__":
    main()

