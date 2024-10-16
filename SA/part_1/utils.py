# Add functions or classes used for data loading and preprocessing
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import Dataset
from pprint import pprint
from functools import partial
import os

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU

# load the data
def load_data(path):
    '''
        input: path/to/data
        output: samples
    '''
    dataset = []
    with open(path) as f:
        for row in f:
          sample = {}
          row = row.split("####")
          sample['utterance'] = row[0]


          # Sentiment
          words = row[1].split() # split the string in row[1] by spaces to separate the words
          sentiments = [word.split('=')[-1] for word in words] # for each word, split it by the '=' sign and take the part after the '='
          sample['sentiment'] = sentiments

          # Extraction of aspects only
          sample['sentiment'] = [s[0] for s in sample['sentiment']]

          # Words
          words = row[1].split() # split the string in row[1] by spaces to separate the words
          cleaned_words = [ # for each word, split it by the '=' sign and
              word.split("=")[0] if word[0] != "=" else word[1:]  # remove the '=' at the start of the word if it exists
              for word in words
          ]
          sample['words'] = cleaned_words

          dataset.append(sample)
    return dataset

# Create Dev Set
def create_dev_set(tmp_train_raw, test_raw, portion = 0.10):
    '''Create a development set from the training set

    Returns:
        train_raw: list of training examples
    
    '''
    # First we get the 10% of the training set, then we compute the percentage of these examples
    sentiments = [s for x in tmp_train_raw for s in set(x['sentiment'])] # extract unique sentiments
    count_y = Counter(sentiments)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(tmp_train_raw):
        if not any(count_y[s] == 1 for s in y['sentiment']): # if none of the sentiments occur exactly once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y['sentiment'])
        else:
            mini_train.append(tmp_train_raw[id_y])  # Add to mini_train if any sentiment occurs once

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    return train_raw, dev_raw, test_raw

# Preprocess the dataset
def preprocess_dataset(train_raw, dev_raw, test_raw):
    '''Preprocess the dataset

    Args:
        train_raw: list of training examples
        dev_raw: list of development examples
        test_raw: list of test examples
        
    Returns:
        sentiments: set of unique sentiments
    '''
    corpus = train_raw + dev_raw + test_raw # We do not want unk labels,

                                                # however this depends on the research purpose
    sentiments = set([sentiment for row in corpus for sentiment in row['sentiment']]) # unique sentiments

    return sentiments

# Create a vocabulary
class Lang():
    '''Class to process the vocabulary and labels
    
    Args:
        words: list of words
        sentiments: list of sentiments
        pad_token_id: int, id of the padding token
        cutoff: int, minimum frequency of words to be included in the vocabulary
    '''
    def __init__(self, sentiments, pad_token_id, cutoff=0):
        #self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.sent2id = self.lab2id(sentiments)
        #self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2sent = {v:k for k, v in self.sent2id.items()}

        self.pad_token_id = pad_token_id
        self.label_pad = 0

    # def w2id(self, elements, cutoff=None, unk=True):
    #     vocab = {'pad': PAD_TOKEN}
    #     if unk:
    #         vocab['unk'] = len(vocab)
    #     count = Counter(elements)
    #     for k, v in count.items():
    #         if v > cutoff:
    #             vocab[k] = len(vocab)
    #     return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = 0
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
# Create dataset class
class DatasetABSA(Dataset):
    '''Dataset class for aspect-based sentiment analysis
    
    Args:
        dataset: list of examples
        lang: Lang object
        unk: str, unknown token
        tokenizer: tokenizer object
    '''
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk', tokenizer=None):
        self.utterances = []
        self.sentiments = []
        self.words = []
        self.lang = lang

        self.tokenizer = tokenizer


        for x in dataset:
            self.utterances.append(x['utterance'])
            self.sentiments.append(x['sentiment'])
            self.words.append(x['words'])

        self.sent_ids = self.mapping_seq(self.sentiments, lang.sent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        words_to_tokenize = ' '.join(self.words[idx])
        tokenized = self.tokenizer(words_to_tokenize, return_tensors='pt') # tokenize the words
        utterance_token = tokenized['input_ids'][0]
        attention_token = tokenized['attention_mask'][0]

        word_ids = tokenized.word_ids() # returns a list where each token is mapped to a specific word in the original sentence.
                                        # If a word is split into multiple subwords, they all get the same word ID.
                                        # 'None' values represent tokens not tied to any word (like special tokens).

        # adjusts the word_ids to account for spaces between words
        # decrement the word id if a word doesn't have a space before it and isn't the first word
        id = 0
        prev_word = None
        for i, w in enumerate(word_ids):
            if w is None or w == 0:
                continue
            char_span = tokenized.word_to_chars(w) # (start and end character positions) of word w in the original sentence
            if words_to_tokenize[char_span[0]-1] != ' ' and prev_word != w: # no space before the word => multiple tokens belong to the same word => increment id
                id += 1
            prev_word = w
            word_ids[i] = w - id

        sentiment = torch.Tensor(self.sent_ids[idx])

        # take only the first word piece of each word, keep the index of the first word piece
        words = set(word_ids)
        words.remove(None)
        mapping_words = torch.Tensor([word_ids.index(i) for i in set(words)])

        if(len(mapping_words) != len(sentiment)):
            assert(f"Length mismatch: mapping_slots has {len(mapping_words)} elements, but sentiment has {len(sentiment)}")

        sample = {'utterance': utterance_token,
                  'attention_mask': attention_token,
                  'mapping_words': mapping_words,
                  'sentence': self.words[idx],
                  'sentiment': sentiment,
                  }

        return sample

    # Auxiliary methods

    def mapping_labels(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                tmp_seq.append(mapper[x])
            res.append(tmp_seq)
        return res
    
# Create a collate function 
from torch.utils.data import DataLoader
def collate_fn(data, lang):
    def merge(sequences, pad_token):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]


    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'], lang.pad_token_id)
    y_sents, y_lengths = merge(new_item["sentiment"],0)
    attention_mask, _ = merge(new_item["attention_mask"],0)
    mapping_words, _ = merge(new_item["mapping_words"],0)

    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_sents = y_sents.to(device)
    attention_mask = attention_mask.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    #mapping_words = mapping_words.to(device)

    new_item["utterances"] = src_utt
    new_item["y_sents"] = y_sents
    new_item["sent_len"] = y_lengths
    new_item["attention_mask"] = attention_mask
    new_item["mapping_words"] = mapping_words

    return new_item

# Create dataset
def create_dataset(train_raw, dev_raw, test_raw, tokenizer, lang):
  train_dataset = DatasetABSA(dataset=train_raw, lang=lang, tokenizer=tokenizer)
  dev_dataset = DatasetABSA(dataset=dev_raw, lang=lang, tokenizer=tokenizer)
  test_dataset = DatasetABSA(dataset=test_raw, lang=lang, tokenizer=tokenizer)

  return train_dataset, dev_dataset, test_dataset

# Create dataloader
def create_dataloader(train_dataset, dev_dataset, test_dataset, lang, batch_size=32):
  collate_fn_with_lang = partial(collate_fn, lang=lang) # We use partial to pass the lang argument to the collate_fn
  train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_with_lang,  shuffle=True)
  dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn_with_lang)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_with_lang)
  return train_loader, dev_loader, test_loader