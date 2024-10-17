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
# from torch.utils.tensorboard import SummaryWriter # tensorboard
import wandb 
import random
import torch.optim as optim

from utils import * # Import all the functions from the utils.py file
from model import LM_RNN, LM_LSTM #TODO


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
def eval_loop(data, model, lang, criterion):
    '''Evaluation loop for the model
    
    Args:
        data: data loader for the evaluation data
        model: model to evaluate
        lang: lang class with the vocabulary     
        criterion: loss function
    
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
def train_and_evaluate(tmp_train_raw, test_raw, lr, runs=1, n_epoch=200, clip=5, patience=5, device='cuda:0', hid_size=200, emb_size=300, model_type='LSTM', optimizer_type='SGD', use_dropout=False):
    '''Running function : preprocess, train and evaluate the model
    
    Args:
        tmp_train_raw: training data
        test_raw: test data
        lr: learning rate
        runs: number of runs
        n_epoch: number of epochs
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
    train_loader, dev_loader, test_loader = create_dataloader(train_dataset, dev_dataset, test_dataset, lang, batch_size=32)
    # end preproces
    
    vocab_len = len(lang.word2id)
    
    #TODO
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    
    
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip=5)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            # Add scalars to TensorBoard
            #writer.add_scalar('Loss/Train', np.asarray(loss).mean(), epoch)
            #writer.add_scalar('PPL/Dev', ppl_dev, epoch)

            # log metrics to wandb
            wandb.log({'Loss/Train': np.asarray(loss).mean(), 'PPL/Dev': ppl_dev, 'epoch': epoch})


            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    wandb.log({'Final PPL/Test': final_ppl})
    wandb.finish() 
    return final_ppl



    