import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class ABSAmodel(BertPreTrainedModel): # Aspect Based Sentiment Analysis model
  '''Model for Aspect Based Sentiment Analysis

  Args:
      BertPreTrainedModel: BERT pretrained model
  '''
  def __init__(self, config, num_labels, dropout_prob=0.1):
    '''
    Args:
        config : BERT configuration
        num_labels : Number of labels (3 for positive, negative, neutral)
        dropout_prob (float, optional): Defaults to 0.1.
    '''
    super(ABSAmodel, self).__init__(config)

    self.num_labels = num_labels
    self.bert = BertModel(config) # pretrained BERT
    self.dropout = nn.Dropout(dropout_prob)

    # 
    self.slot_out = self.slot_out = nn.Linear(self.config.hidden_size, num_labels) # slot filling head

  def forward(self, token_ids, attention_mask, mapping_words):
    '''Forward pass of the model

    Args:
        token_ids: 
        attention_mask : attention mask for the input
        mapping_words : mapping of words to their corresponding indices in the sentence

    Returns:
        slot_logits : logits for each word in the sentence
    '''    
    out = self.bert(token_ids, attention_mask=attention_mask) # BERT outputs

    sequence_output = out[0]  # [batch_size, seq_length, hidden_size]

    # apply dropout
    sequence_output = self.dropout(sequence_output)

    # extract the embeddings of the first token corresponding to each word in the sentence
    slot = []
    for i in range(sequence_output.size(0)):
        indices = mapping_words[i]
        selected = sequence_output[i, indices, :]
        slot.append(selected)
    slot = torch.stack(slot)

    # Compute slot logits for each word
    slot_logits = self.slot_out(slot)  # [batch_size, num_words, slots]
    slot_logits = slot_logits.permute(0, 2, 1)  # to compute the cross-entropy loss


    return slot_logits