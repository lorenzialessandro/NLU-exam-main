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
from model import LM_LSTM

# Non-monotonically Triggered AvSGD (NT-AvSGD)
class NTAvSGD(optim.SGD):
    '''Non-monotonically Triggered AvSGD (NT-AvSGD) optimizer
    
    Args:
        model: model to optimize
        dev_loader: data loader for the validation set
        lang: lang class with the vocabulary
        stop_criterion: stopping criterion
        lr: learning rate
        L: logging interval
        n: non-monotone interval
    '''
    def __init__(self, model, dev_loader, lang, stop_criterion, lr, L=200, n=5):
        super(NTAvSGD, self).__init__(model.parameters(), lr=lr)
        self.dev_loader = dev_loader
        self.model = model
        self.lang = lang
        self.stop_criterion = stop_criterion
        self.lr = lr
        self.L = L
        self.n = n
        
        self.k = 0 #
        self.T = 0
        self.t = 0
        self.logs = []
        self.tmp = {} # Temporary storage for the model parameters
        self.avg = {} # Average of the model parameters
        
    def step(self, closure=None):
        super(NTAvSGD, self).step(closure) # Compute stochastic gradient ∇ f(wk) and take SGD step (1)
        
        with torch.no_grad():
            # Every L iterations, evaluate the model on the validation set
            if self.k % self.L == 0 and self.T == 0:
                v, _ = eval_loop(self.dev_loader, self.model, self.lang, self.stop_criterion) # Compute validation perplexity v
                self.model.train() # Set the model to train mode after evaluation
                
                # Check the non-monotonic condition: if enough iterations have passed and the current loss is greater than the minimum loss in the last n iterations
                if self.t > self.n and self.logs[-1] > min(self.logs[-self.n:]):
                    self.T = self.k # Non-monotonic condition is met, set T = k
                
                self.logs.append(v) 
                self.t += 1
                
        self.k += 1
        
        if self.T > 0:
            for param in self.model.parameters():
                if param not in self.tmp:
                    self.avg[param] = param.data.clone() # Initialize the average with the current parameters
                else:
                    self.avg[param] = self.avg[param] + (param.data - self.avg[param]) / (self.k - self.T + 1) # Update the average
            
    def reset(self):
        '''Reset the parameters of the model to the average'''
        if self.T > 0:
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data = self.tmp[param].clone() # Reset the model parameters
                    
    def average_parameters(self):
        '''Average the parameters of the model'''
        if self.T > 0:
            with torch.no_grad():
                for param in self.model.parameters():
                    #if param not in self.tmp:
                    self.tmp[param] = param.data.clone() # Initialize the temporary storage with the current parameters
                    param.data = self.avg[param].clone() # Set the model parameters to the average
            

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
def run(train_raw, dev_raw, test_raw, lr, runs=1, epochs=200, clip=5, patience=5, device='cuda:0', hid_size=300, emb_size=300, optimizer_type='SGD', weight_tying=False, var_dropout=False):
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
        hid_size: size of the hidden layer (default is 300)
        emb_size: size of the embedding layer (default is 300)
        optimizer_type: type of the optimizer (default is SGD)
        weight_tying: use weight tying (default is False)
        var_dropout: use variational dropout (default is False)
    '''
    
    # preprocess : create the datasets, dataloaders, lang class, optimizer and the model
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"]) # Vocab is computed only on training set
    
    lang = Lang(train_raw, ["<pad>", "<eos>"]) # We add two special tokens end of sentence and padding
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    train_dataset, dev_dataset, test_dataset = create_dataset(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = create_dataloader(train_dataset, dev_dataset, test_dataset, lang, batch_size=256)
    # end preprocess
    
    vocab_len = len(lang.word2id)
    
    model = None
    optimizer = None
    best_model_runs = None
    ppls = []
    best_ppl_runs = math.inf
    use_nta = False # Use NT-AvSGD
    
    # start the runs
    for x in tqdm(range(0, runs)):  
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], weight_tying=weight_tying, var_dropout=var_dropout).to(device)
        model.apply(init_weights)
        # Optimizer selection
        if optimizer_type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_type == "NTAvSGD":
            optimizer = NTAvSGD(model, dev_loader, lang, criterion_eval, lr, L=epochs, n=5)
            use_nta = True
        else:
            print("Optimizer not implemented")
            return
        
        # start the run
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        best_mode_avg = None # Best model with averaged parameters
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

                # log metrics to wandb
                wandb.log({'Loss/Train': np.asarray(loss).mean(), 'PPL/Dev': ppl_dev, 'epoch': epoch})

                if use_nta: # Use NT-AvSGD
                    optimizer.average_parameters()
                    ppl_dev, loss_dev = eval_loop(dev_loader, model, lang, criterion_eval)
                    optimizer.reset()
                    # log metrics to wandb
                    wandb.log({'PPL/Dev/NTA': ppl_dev, 'epoch': epoch})
                    
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    
                    if use_nta: # Use NT-AvSGD
                        optimizer.average_parameters()
                        best_model_avg = copy.deepcopy(model).to('cpu')
                        
                        optimizer.reset()
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
        
        if use_nta: # Use NT-AvSGD
            best_model_avg.to(device)
            final_ppl_avg, _ = eval_loop(test_loader, best_model_avg, lang, criterion_eval)
            best_model_avg.to("cpu")
            print('Test ppl/NTA: ', final_ppl_avg)
            wandb.log({'Final PPL/Test/NTA': final_ppl_avg})
        
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


    