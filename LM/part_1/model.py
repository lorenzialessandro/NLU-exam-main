import torch
import torch.nn as nn

# Recurrent Neural Networks (RNN)
class LM_RNN(nn.Module):
    '''RNN model for language modeling'''
    
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        '''RNN model for language modeling
    
        Args:
            emb_size (int): embedding size
            hidden_size (int): hidden size
            output_size (int): output size
            pad_index (int): padding index
            out_dropout (float): dropout probability for the output layer
            emb_dropout (float): dropout probability for the embedding layer
            n_layers (int): number of layers
        '''
        
        super(LM_RNN, self).__init__()
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        '''Forward pass
        
        Args:   
            input_sequence (torch.Tensor): input sequence tensor
        
        Returns:
            output (torch.Tensor): output tensor
        '''
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    
# Long-Short-Term-Memories (LSTM)
class LM_LSTM(nn.Module):
    '''LSTM model for language modeling'''
    
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, use_dropout=False, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        '''LSTM model for language modeling
        
        Args:
            emb_size (int): embedding size
            hidden_size (int): hidden size
            output_size (int): output size
            pad_index (int): padding index
            use_dropout (bool): use dropout layer
            out_dropout (float): dropout probability for the output layer
            emb_dropout (float): dropout probability for the embedding layer
            n_layers (int): number of layers
        '''
            
        super(LM_LSTM, self).__init__()
        
        self.use_dropout = use_dropout
        
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        if self.use_dropout:
            self.emb_dropout = nn.Dropout(emb_dropout) # dropout layer after embedding
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(out_dropout) # dropout layer before the linear layer
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        '''Forward pass
        
        Args:
            input_sequence (torch.Tensor): input sequence tensor
        
        Returns:
            output (torch.Tensor): output tensor
        '''
        emb = self.embedding(input_sequence)
        if self.use_dropout:
            emb = self.emb_dropout(emb) # dropout after embedding layer
        rnn_out, _  = self.rnn(emb)
        if self.use_dropout:
            rnn_out = self.dropout(rnn_out) # dropout before the output layer
        output = self.output(rnn_out).permute(0,2,1)
        return output