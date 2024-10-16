import torch
import torch.nn as nn
import torch.optim as optim

# =============== LSTM (dropout) with weight tying ===============
    # shares the weights between the embedding and softmax layer
    # number of paramters of the model is reduced
  
    # => Reduced Parameters: tying the weights the model has fewer parameters to learn. This can lead to faster training times and reduced memory requirements.

    # => Improved Generalization: tying the weights can encourage the model to learn more meaningful representations since it's constrained to learn embeddings that are useful both for input and output.

class LM_LSTM_weight_tying(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.4,
                 emb_dropout=0.4, n_layers=1):
        super(LM_LSTM_weight_tying, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(out_dropout)
        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size, bias=False) # embedding weights for the output layer
          # bias=False to remove any additional parameters that might introduce unnecessary complexity
        self.output.weight = self.embedding.weight # weight tying

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb)
        rnn_out = self.dropout(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
    
    
    
# =============== Variational Dropout Class ===============
    # samples a binary dropout mask only once upon the first call and then to repeatedly use that (so not every time the dropout function is called)
    # locked dropout mask for all repeated connections within the forward and backward pass
    # Variational Dropout ensures consistency in the dropout pattern over time, enabling better learning of temporal dependencies
class VarDropout(nn.Module):
  def __init__(self, dropout):
    super(VarDropout, self).__init__()
    self.dropout = dropout

  def forward(self, input_tensor):

        # if the module is in training mode => generate a mask
        # mask is sampled from a Bernoulli distribution with p: 1 - dropout
        # mask is applied element-wise to the input tensor
        # masked tensor is scaled to ensure that the expected value of the tensor remains the same during training and inference

        if not self.training:
            return input_tensor

        # mask = torch.empty_like(input_tensor).bernoulli_(1 - self.dropout)
        mask = torch.bernoulli(torch.full_like(input_tensor, 1 - self.dropout))
        
        dropped_out = mask * input_tensor

        scaled = dropped_out / (1 - self.dropout)

        return scaled
    
    
    
# =============== LSTM with variational dropout (weight tying) ===============
    # the dropout mask is applied to the embedding layer and to the LSTM layer  
    # helps prevent overfitting in LSTMs and improve generalization performance
class LM_LSTM_VariationalDropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.4, emb_dropout=0.4, rnn_dropout = 0.4, n_layers=1):
        super(LM_LSTM_VariationalDropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        self.dropout_emb = VarDropout(emb_dropout) # variational dropout
        self.dropout_rnn = VarDropout(rnn_dropout) # variational dropout

        self.output = nn.Linear(hidden_size, output_size, bias=False) # embedding weights for the output layer
          # bias=False to remove any additional parameters that might introduce unnecessary complexity
        self.output.weight = self.embedding.weight # weight tying

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout_emb(emb)
        rnn_out, _  = self.rnn(emb)
        rnn_out = self.dropout_rnn(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
    
    
# =============== =============== ===============

# Optimizer

# =============== Non-monotonically Triggered AvSGD (NT-AvSGD) ===============
    # non-monotonically triggered variant of AvSGD (NT-AvSGD)
    # uses a constant learning rate 
    # incorporate a non-monotonic condition for triggering updates to the acceleration parameters

class NTAvSGD(optim.Optimizer):
    def __init__(self, params, lr, total_samples, batch_size, n=5, weight_decay=0): 
        # lr    learning rate
        # n     non-monotone interval
        # L     logging interval
        
        # set L to be the number of iterations in an epoch and n = 5 (per paper https://arxiv.org/abs/1708.02182)
        L = total_samples // batch_size
        
        defaults = dict(lr=lr, L=L, n=n, weight_decay=weight_decay, T=0, t=0, logs=[], loss=None) 
        super(NTAvSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i in group['params']:
                grad = i.grad.data
                state = self.state[i]

                if len(state) == 0:
                    state['k'] = 0
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(i.data)
                    
                if group['T'] == 0: # if mod(k,L) == 0 and T = 0: 
                    if (state['k'] % group['L'] == 0 and group['t'] > group['n'] and (group['loss'] is not None and min(group['logs'][:-group['n']]) is not None) and group['loss'] > min(group['logs'][:-group['n']])): # min l∈{0,··· ,t−n−1} logs[l]
                        group['T'] = state['k']

                state['k'] += 1
                group['logs'].append(loss)
                group['t'] += 1

                # update
                i.data.add_(grad, alpha=-group['lr']) # i.data = i.data + (-lr * grad)

                if state['mu'] == 1:
                    state['ax'].copy_(i.data)
                else:
                    # update
                    diff = (i.data.sub(state['ax'])).mul(state['mu'])
                    state['ax'].add_(diff) # (i.data - ax) * mu

                # update
                state['mu'] = 1 / max(1, state['k'] - group['T'])

        return loss