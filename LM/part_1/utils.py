# Add functions or classes used for data loading and preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
from functools import partial
from torch.utils.data import DataLoader

device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU

# prepares the text data by adding end-of-sentence tokens
def read_file(path, eos_token="<eos>"):
    '''Reads a file and adds an end-of-sentence token to each line
    
    Args:
        path (str): the path to the file
        eos_token (str): the end-of-sentence token to add
    
    
    Returns:
        output: a list of strings, each representing a line from the file with an end-of-sentence token
    '''
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    '''Creates a vocabulary mapping each word to a unique index, including special tokens if provided
    
    Args:
        corpus (list): a list of strings, each representing a sentence
        special_tokens (list): a list of special tokens to include in the vocabulary
    
    Returns:
        output: a dictionary mapping words to indices
    '''
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# Create a vocabulary
class Lang():
    '''Create a vocabulary for a given corpus of text data
    
    Args:
        corpus (list): a list of strings, each representing a sentence
        special_tokens (list): a list of special tokens to include in the vocabulary
    '''    
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    
 
# Create dataset class    
class PennTreeBank (data.Dataset):
    '''Dataset class for the Penn Treebank dataset
    
    Args:
        corpus (list): a list of strings, each representing a sentence
        lang (Lang): a Lang object created from the corpus
    '''
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

# Create a collate function
def collate_fn(data, pad_token):
    def merge(sequences):
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
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

# Create dataset
def create_dataset(train_raw, dev_raw, test_raw, lang):
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    return train_dataset, dev_dataset, test_dataset

# Create dataloader
def create_dataloader(train_dataset, dev_dataset, test_dataset, lang, batch_size=32):
  collate_fn_with_lang = partial(collate_fn, pad_token=lang.word2id["<pad>"])
  train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_with_lang, shuffle=True)
  dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn_with_lang)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_with_lang)
  return train_loader, dev_loader, test_loader

    

# =============== =============== ===============

def preprocess_and_load_data():
    # =============== Reading and Loading Data ===============
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")


    # =============== Vocabulary Preparation ===============
    # Vocab is computed only on training set
    # We add two special tokens end of sentence and padding
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # =============== Dataset Preparation ===============
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # =============== Dataloader instantiation ===============
    # 256, 1024, 1024
    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    return train_loader, dev_loader, test_loader, lang