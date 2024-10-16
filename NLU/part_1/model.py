import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Model for Intent and Slot Filling Task
class ModelIAS(nn.Module):
    '''Model for Intent and Slot Filling Task'''
    
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, bidirectionality, dropout_layer, n_layer=1, pad_index=0):
        '''Model for Intent and Slot Filling Task
        
        Args:
            hid_size (int): Hidden size
            out_slot (int): number of slots (output size for slot filling)
            out_int (int): number of intents (output size for intent class)
            emb_size (int): word embedding size
            vocab_len (int): vocabulary size
            bidirectionality (bool): True means adding bidirectionality to the model
            dropout_layer (bool): True means adding dropout layer to the model
            n_layer (int): number of layers for LSTM
            pad_index (int): padding index for the embedding layer
        '''
        
        super(ModelIAS, self).__init__()
                
        self.bidirectionality = bidirectionality
        self.dropout_layer = dropout_layer

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectionality, batch_first=True)
        
        if not self.bidirectionality:
          self.slot_out = nn.Linear(hid_size, out_slot)
          self.intent_out = nn.Linear(hid_size, out_int)
        else:
          self.slot_out = nn.Linear(hid_size * 2, out_slot)  # Double the hid_size for bidirectional LSTM
          self.intent_out = nn.Linear(hid_size * 2, out_int)  # Double the hid_size for bidirectional LSTM
        
        self.dropout = nn.Dropout(0.5) # add the dropout layer with a dropout probability of 0.5

    def forward(self, utterance, seq_lengths):
        '''Forward pass
        
        Args:
            utterance (torch.Tensor): input utterance tensor
            seq_lengths (torch.Tensor): sequence lengths tensor
          
        Returns:
            slots (torch.Tensor): slot logits
            intent (torch.Tensor): intent logits
        '''
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        if self.dropout_layer:
          # apply dropout to embeddings
          utt_emb = self.dropout(utt_emb)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.dropout_layer:
          # apply dropout to the unpacked sequence
          utt_encoded = self.dropout(utt_encoded)
       
        if not self.bidirectionality:
          # Get the last hidden state
          last_hidden = last_hidden[-1,:,:]
        else:
          # Concatenation of the forward and backward hidden states along the feature dimension (dim=1)
          # to obtain the final hidden state for both slots and intent prediction
          last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1) 

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent