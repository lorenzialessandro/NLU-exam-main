# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math
import torch
import torch.nn as nn
import math
import numpy as np
import copy
from copy import deepcopy
from tqdm import tqdm
import wandb 
import random
import torch.optim as optim

from utils import * # Import all the functions from the utils.py file
from model import LM_RNN, LM_LSTM


# Training loop
def train_loop(data, optimizer, model, lang, criterion, clip=5):
    '''Train loop for the model
    
    Args:
        data: data loader for the training data
        optimizer: optimizer to use
        model: model to train 
        lang: lang class with the vocabulary     
        criterion: loss function
        clip: gradient clipping (default is 5)
        
    Returns:
        loss: average loss for the epoch
    '''
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

# Evaluation loop
def eval_loop(data, model, lang, eval_criterion):
    '''Evaluation loop for the model
    
    Args:
        data: data loader for the evaluation data
        model: model to evaluate
        lang: lang class with the vocabulary     
        eval_criterion: loss function
    
    Returns:
        ppl: perplexity of the model
    '''
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

# Initializes the weights of the model. It is called during model initialization to set the initial weights
def init_weights(mat):
    '''Initialize the weights of the model'''
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
                    
                    
# Running the training and evaluation loops             
def run(train_raw, dev_raw, test_raw, lr, runs=1, epochs=200, clip=5, patience=5, device='cuda:0', hid_size=200, emb_size=300, model_type='LSTM', optimizer_type='SGD', use_dropout=False):
    '''Running function : preprocess, train and evaluate the model
    
    Args:
        train_raw: training data
        dev_raw: dev data
        test_raw: test data
        lr: learning rate
        runs: number of runs
        epochs: number of epochs
        clip: gradient clipping (default is 5)
        patience: patience for early stopping
        device: device to use (default is cuda:0)
        hid_size: size of the hidden layer (default is 200)
        emb_size: size of the embedding layer (default is 300)
        model_type: type of the model (default is LSTM)
        optimizer_type: type of the optimizer (default is SGD)
        use_dropout: use dropout layer (default is False)
    '''
    
    # preprocess : create the datasets, dataloaders, lang class, optimizer and the model
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"]) # Vocab is computed only on training set
    
    lang = Lang(train_raw, ["<pad>", "<eos>"]) # We add two special tokens end of sentence and padding
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    train_dataset, dev_dataset, test_dataset = create_dataset(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = create_dataloader(train_dataset, dev_dataset, test_dataset, lang, batch_size=256)
    # end preproces
    
    vocab_len = len(lang.word2id)
    
    model = None
    optimizer = None
    best_model_runs = None
    ppls = []
    best_ppl_runs = math.inf
    best_model_runs = None
    
    # start the runs
    for x in tqdm(range(0, runs)):
        # Model selection
        if model_type == "RNN":
            model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        elif model_type == "LSTM":
            model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], use_dropout=use_dropout).to(device)
        else:
            print("Model not implemented")
            return
        
        # Optimizer selection
        if optimizer_type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        else:
            print("Optimizer not implemented")
            return
        
        # start the run
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        patience_p = patience
        
        pbar = tqdm(range(1,epochs))
        
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, model, lang, criterion_train, clip=5)
            if epoch % 1 == 0: # We check the performance every 1 epoch
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                
                ppl_dev, loss_dev = eval_loop(dev_loader, model, lang, criterion_eval)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)

                # Add scalars to TensorBoard
                #writer.add_scalar('Loss/Train', np.asarray(loss).mean(), epoch)
                #writer.add_scalar('PPL/Dev', ppl_dev, epoch)

                # log metrics to wandb
                wandb.log({'Loss/Train': np.asarray(loss).mean(), 'PPL/Dev': ppl_dev, 'epoch': epoch})

                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    # save the model
                    best_model = copy.deepcopy(model).to('cpu')
                    patience_p = patience # reset to patience
                else:
                    patience_p -= 1
                if patience_p <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean
                
        if best_model is None:
            best_model = model

        best_model.to(device)
        final_ppl,  _ = eval_loop(test_loader, best_model, lang, criterion_eval)
        best_model.to("cpu")
        
        ppls.append(final_ppl)
        
        #print('Test ppl: ', final_ppl)
        wandb.log({'Final PPL/Test': final_ppl})
         
        if final_ppl < best_ppl_runs:
            best_ppl_runs = final_ppl
            #best_model_runs = copy.deepcopy(best_model)
            #show plot
            # plt.plot(sampled_epochs, losses_train, label='Train')
            # plt.plot(sampled_epochs, losses_dev, label='Dev')
            # plt.legend()
            # plt.title(f"Loss {test_name}")
            # plt.show()
            
    ppls = np.asarray(ppls)
    
    wandb.log({"PPL": round(ppls.mean(),3)})
    print('PPL', round(ppls.mean(),3), '+-', round(ppls.std(),3))
    
    # Save the model
    #path = 'model_bin/LMModel.pt'
    #torch.save(best_model.state_dict(), path)



    