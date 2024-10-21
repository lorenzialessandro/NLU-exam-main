import torch
import torch.nn as nn
import torch.optim as optim

class VarDropout(nn.Module):
    '''Variational dropout for the embedding and LSTM layers'''
    
    def __init__(self, dropout):
        '''Variational dropout for the embedding and LSTM layers
        
        Args:
            dropout (float): dropout probability
        '''
        super(VarDropout, self).__init__()
        self.dropout = dropout

    def forward(self, input_tensor):
        '''Forward pass
        
        Args:
            input_tensor (torch.Tensor): input tensor
        
        Returns:
            scaled (torch.Tensor): scaled tensor
        '''
        if not self.training or self.dropout == 0:
            return input_tensor
        
        batch_size, _, feature_size = x.size() # get batch size, sequence length, feature size
        
        mask = torch.bernoulli(torch.full((batch_size, 1, feature_size), 1 - self.dropout, device=x.device)) # generate mask
        
        scaled = input_tensor.mul(mask).div(1 - self.dropout) # apply mask and scale by (1 - dropout)

        return scaled

# Long-Short-Term-Memories (LSTM)
class LM_LSTM(nn.Module):
    '''LSTM model for language modeling'''
    
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, weight_tying=False, var_dropout=False, out_dropout=0.3,
                 emb_dropout=0.3, n_layers=1):
        '''LSTM model for language modeling
        
        Args:
            emb_size (int): embedding size
            hidden_size (int): hidden size
            output_size (int): output size
            pad_index (int): padding index
            weight_tying (bool): use weight tying 
            var_dropout (bool): use variational dropout
            out_dropout (float): dropout probability for the output layer
            emb_dropout (float): dropout probability for the embedding layer
            n_layers (int): number of layers
        '''
        
        super(LM_LSTM, self).__init__()
        
        self.weight_tying = weight_tying
        print(self.weight_tying)
        self.var_dropout = var_dropout
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        self.pad_token = pad_index
        
        if self.var_dropout:
            self.dropout_emb = VarDropout(emb_dropout) # variational dropout
            self.dropout_rnn = VarDropout(rnn_dropout) # variational dropout

        self.output = nn.Linear(hidden_size, output_size)
        if self.weight_tying:
            self.output.weight = self.embedding.weight # weight tying

    def forward(self, input_sequence):
        '''Forward pass
        
        Args:
            input_sequence (torch.Tensor): input sequence tensor
        
        Returns:
            output (torch.Tensor): output tensor
        '''
        emb = self.embedding(input_sequence)
        if self.var_dropout:
            emb = self.dropout_emb(emb)
        rnn_out, _  = self.rnn(emb)
        if self.var_dropout:
            rnn_out = self.dropout_rnn(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output