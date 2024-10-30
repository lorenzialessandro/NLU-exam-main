# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math
import numpy as np
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertTokenizer, BertModel, BertTokenizerFast, BertPreTrainedModel
import matplotlib.pyplot as plt
from copy import deepcopy
import wandb
import random
import torch.optim as optim

from utils import * # Import all the functions from the utils.py file
from model import ABSAmodel # Import the model from the model.py file

# adapted from https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py
def match_ts(gold_ts_sequence, pred_ts_sequence, lang):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    #tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    # need to adapt this :
    tag2tagid = lang.sent2id # {'pad': 0, 'O': 1, 'T': 2}

    n_labels = len(lang.sent2id)-1 # adapted to replace the '3' with n_labels

    hit_count, gold_count, pred_count = np.zeros(n_labels), np.zeros(n_labels), np.zeros(n_labels)

    # need to rewrite this part (assumes a specific structure for t[2])
    # for t in gold_ts_sequence:
    #     #print(t)
    #     ts_tag = t[2]
    #     tid = tag2tagid[ts_tag]
    #     gold_count[tid] += 1
    # for t in pred_ts_sequence:
    #     ts_tag = t[2]
    #     tid = tag2tagid[ts_tag]
    #     if t in gold_ts_sequence:
    #         hit_count[tid] += 1
    #     pred_count[tid] += 1

    #  For each element in gold_ts_sequence, the corresponding element in pred_ts_sequence is compared.
    # If they are equal, the hit_count is incremented for the corresponding tag ID
    for i in range(len(gold_ts_sequence)):
        if gold_ts_sequence[i] == pred_ts_sequence[i]:
            hit_count[tag2tagid[gold_ts_sequence[i]]-1] += 1
        gold_count[tag2tagid[gold_ts_sequence[i]]-1] += 1
        pred_count[tag2tagid[pred_ts_sequence[i]]-1] += 1
    return hit_count, gold_count, pred_count
  
  
# adapted from https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py
SMALL_POSITIVE_CONST = 1e-4
def evaluate_ts(gold_ts, pred_ts, lang):
    #print('gold_ts: ', len(gold_ts))
    #print('pred_ts: ', len(pred_ts))
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return:
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)

    n_labels = len(lang.sent2id)-1 # adapted to replace the '3' with n_labels
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(n_labels), np.zeros(n_labels), np.zeros(n_labels)
    ts_precision, ts_recall, ts_f1 = np.zeros(n_labels), np.zeros(n_labels), np.zeros(n_labels)

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        #g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(ts_tag_sequence=p_ts)
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts,
                                                              pred_ts_sequence=p_ts, lang=lang)

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        # calculate macro-average scores for ts task
    for i in range(n_labels):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    # calculate micro-average scores for ts task
    n_tp_total = sum(n_tp_ts)
    # total sum of TP and FN
    n_g_total = sum(n_gold_ts)
    # total sum of TP and FP
    n_p_total = sum(n_pred_ts)

    ts_macro_f1 = ts_f1.mean() # F1 macro
    ts_micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST) # Precision
    ts_micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST) # Recall
    ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + SMALL_POSITIVE_CONST) # F1 Micro

    ts_scores = {"F1 macro":ts_macro_f1, "precision":ts_micro_p, "recall":ts_micro_r, "F1 micro":ts_micro_f1}
    return ts_scores
  
# Training loop
def train_loop(data, optimizer, model, lang, criterion_sentiments, clip=5):
    '''Train loop for the model

    Args:
        data: data loader for the training data
        optimizer: optimizer to use
        model: model to train 
        lang: lang class with the vocabulary     
        criterion_sentiments: loss function
        clip: gradient clipping (default is 5)

    Returns:
        loss_array: array with the loss values for each batch
    '''

    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient

        sentiments = model(sample['utterances'], sample['attention_mask'], sample['mapping_words'])

        # calculate the loss on the sentiments
        loss = criterion_sentiments(sentiments, sample['y_sents'])

        loss_array.append(loss.item())

        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    # print(f"train_loop > loss_array ", loss_array)
    return loss_array

# Evaluation loop
def eval_loop(data, model, lang, criterion_sentiments):
    '''Evaluation loop for the model
    
    Args:
        data: data loader for the evaluation data
        model: model to evaluate
        lang: lang class with the vocabulary
        criterion_sentiments: loss function
    
    Returns:
        results: dictionary with the F1 macro, precision, recall and F1 micro
    '''
    
    model.eval()
    loss_array = []

    ref_sentiments = []
    hyp_sentiments = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            sentiments = model(sample['utterances'], sample['attention_mask'], sample['mapping_words'])
            #print(sentiments.shape)
            # calculate the loss on the sentiments
            loss = criterion_sentiments(sentiments, sample['y_sents'])

            loss_array.append(loss.item())

            # Sentiment inference
            ref_sentiment = []
            hyp_sentiment = []
            output_slots = torch.argmax(sentiments, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['sent_len'][id_seq].tolist()

                gt_ids = sample['y_sents'][id_seq].tolist()
                gt_sentiments = [lang.id2sent[elem] for elem in gt_ids[:length]]

                to_decode = seq[:length].tolist()

                # compute the reference sentiments
                ref_sentiment.append(gt_sentiments)

                # compute the hypothesis sentiments
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append(lang.id2sent[elem])
                hyp_sentiment.append(tmp_seq)

            ref_sentiments.extend(ref_sentiment)
            hyp_sentiments.extend(hyp_sentiment)

    # Compute the F1 score for the slots
    try:
      results = evaluate_ts(ref_sentiments, hyp_sentiments, lang)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("\nWarning:", ex)
        ref_s = set([x[1] for x in ref_sentiments])
        hyp_s = set([x[1] for x in hyp_sentiments])
        print(ref_s)
        print(hyp_s)
        print(hyp_s.difference(ref_s))
        results = {"F1 macro":0, "precision":0, "recall":0, "F1 micro":0}
        
    return results, loss_array

# =============================================================================

# Running the training and evaluation loops
def run(tmp_train_raw, test_raw, bert_model, lr, runs=1, n_epochs=200, clip=5, patience=5, device='cuda:0'):
    '''Running function : preprocess, train and evaluate the model

    Args:
        tmp_train_raw: training data
        test_raw: test data
        bert_model: bert model to use
        lr: learning rate
        runs: number of runs
        n_epochs: number of epochs
        clip: gradient clipping
        patience: patience for early stopping
        device: device to use
    '''
    # preprocess : create the datasets, dataloaders, lang class and the model
    train_raw, dev_raw, test_raw = create_dev_set(tmp_train_raw, test_raw, portion = 0.10)
    sentiments = preprocess_dataset(train_raw, dev_raw, test_raw)
    
    tokenizer = BertTokenizerFast.from_pretrained(bert_model) # Download the tokenizer

    lang = Lang(sentiments, tokenizer.pad_token_id)

    criterion_sentiments = nn.CrossEntropyLoss(ignore_index=0)

    train_dataset, dev_dataset, test_dataset = create_dataset(train_raw, dev_raw, test_raw, tokenizer, lang)
    train_loader, dev_loader, test_loader = create_dataloader(train_dataset, dev_dataset, test_dataset, lang, batch_size=32)
    # end preprocess

    num_labels = len(lang.sent2id)

    f1_macro, precision, recall, f1_micro = [], [], [], []
    best_f1_runs = 0
    best_model_runs = None
    
    # start the runs
    for x in tqdm(range(0, runs)):
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = num_labels  # Add the num_labels parameter to the configuration
        model = ABSAmodel(config=config, num_labels=num_labels).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None
        patience_p = patience

        for e in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, model, lang, criterion_sentiments)
            if e % 5 == 0:
                sampled_epochs.append(e)
                losses_train.append(np.asarray(loss).mean())

                results_dev, loss_dev = eval_loop(dev_loader, model, lang, criterion_sentiments)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['F1 macro']

                # print(f"Intent Acc: {intent_res['accuracy']:.3f} Slot F1: {results_dev['total']['f']:.3f}")
                if f1 > best_f1:
                    best_f1 = f1
                    # save the model
                    best_model = deepcopy(model).to("cpu")
                    patience_p = patience # reset to patience
                else:
                    patience_p -= 1
                if patience_p <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean
        if best_model is None:
            best_model = model

        best_model.to(device)
        results_test, _ = eval_loop(test_loader, best_model, lang, criterion_sentiments)
        best_model.to("cpu")

        f1_macro.append(results_test['F1 macro'])
        precision.append(results_test['precision'])
        recall.append(results_test['recall'])
        f1_micro.append(results_test['F1 micro'])

        # wandb.log({"F1 macro": results_test['F1 macro'], "precision": results_test['precision'], "recall": results_test['recall'], "F1 micro": results_test['F1 micro']})

        # print('F1 macro: ', results_test['F1 macro'])
        # print('Precision: ', results_test['precision'])
        # print('Recall: ', results_test['recall'])
        # print('F1 micro: ', results_test['F1 micro'])

        if results_test['F1 macro'] > best_f1_runs:
            best_f1_runs = results_test['F1 macro']
            #best_model_runs = deepcopy(best_model)
            #show plot
            # plt.plot(sampled_epochs, losses_train, label='Train')
            # plt.plot(sampled_epochs, losses_dev, label='Dev')
            # plt.legend()
            # plt.title(f"Loss {test_name}")
            # plt.show()

    f1_macro = np.asarray(f1_macro)
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    f1_micro = np.asarray(f1_micro)

    print('F1 macro: ', round(f1_macro.mean(),3), '+-', round(f1_macro.std(),3))
    print('Precision: ', round(precision.mean(),3), '+-', round(precision.std(),3))
    print('Recall: ', round(recall.mean(),3), '+-', round(recall.std(),3))
    print('F1 micro: ', round(f1_micro.mean(),3), '+-', round(f1_micro.std(),3))

    # wandb.log({"F1 macro": f1_macro.mean(), "precision": precision.mean(), "recall": recall.mean(), "F1 micro": f1_micro.mean()})

    # Save the model
    # path = 'model_bin/ABSAmodel.pt'
    # torch.save(best_model.state_dict(), path)
    