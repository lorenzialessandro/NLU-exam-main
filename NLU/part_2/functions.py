# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math
import numpy as np
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from conll import evaluate
from sklearn.metrics import classification_report
from transformers import BertTokenizer, AutoTokenizer
import matplotlib.pyplot as plt
from copy import deepcopy
import wandb
import random
import torch.optim as optim

from utils import * # Import all the functions from the utils.py file
from model import JointModel # Import the model from the model.py file

# Training loop
def train_loop(data, optimizer, model, lang, criterion_intents, criterion_slots, clip=5):
    '''Train loop for the model

    Args:
        data: data loader for the training data
        optimizer: optimizer to use
        model: model to train 
        lang: lang class with the vocabulary     
        criterion_intents: loss function for the intents
        criterion_slots: loss function for the slots
        clip: gradient clipping (default is 5)

    Returns:
        loss_array: array with the loss values (loss_intent + loss_slot) for each batch
    '''
    
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient

        slots, intents = model(sample['utterances'], sample['attention_mask'], sample['mapping_slots'])

        # calculate the loss on the intents
        loss_intent = criterion_intents(intents, sample['intents'])
        # calculate the loss on the slots
        loss_slot = criterion_slots(slots, sample['y_slots'])

        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                    # Is there another way to do that?

        loss_array.append(loss.item())

        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    # print(f"train_loop > loss_array ", loss_array)
    return loss_array

# Evaluation loop
def eval_loop(data, model, lang, criterion_intents, criterion_slots):
    '''Evaluation loop for the model
    
    Args:
        data: data loader for the evaluation data
        model: model to evaluate
        lang: lang class with the vocabulary
        criterion_intents: loss function for the intents
        criterion_slots: loss function for the slot
    
    Returns:
        results: F1 score for the slots
        report_intent: classification report for the intents
        loss_array: array with the loss values (loss_intent + loss_slot) for each batch
    '''
    
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['attention_mask'], sample['mapping_slots'])

            # calculate the loss on the intents
            loss_intent = criterion_intents(intents, sample['intents'])
            # calculate the loss on the slots
            loss_slot = criterion_slots(slots, sample['y_slots'])

            loss = loss_intent + loss_slot # In joint training we sum the losses.
                                        # Is there another way to do that?
            loss_array.append(loss.item())

            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            ref_slot = []
            hyp_slot = []
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'][id_seq].tolist()
                utterance = sample['sentence'][id_seq]

                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]

                to_decode = seq[:length].tolist()

                # compute the reference slots
                ref_slot.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])

                # compute the hypothesis slots
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slot.append(tmp_seq)
            ref_slots.extend(ref_slot)
            hyp_slots.extend(hyp_slot)

    # Compute the F1 score for the slots
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("\nWarning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    # Compute the classification report for the intents
    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)

    return results, report_intent, loss_array


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
    slots, intents = preprocess_dataset(train_raw, dev_raw, test_raw)
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model) # Download the tokenizer
    
    lang = Lang(intents, slots, tokenizer.pad_token_id)
    
    criterion_slots = nn.CrossEntropyLoss(ignore_index=lang.pad_token_id)
    criterion_intents = nn.CrossEntropyLoss()
    # end preprocess
    
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    slot_f1s, intent_acc = [], []
    best_f1_runs = 0
    best_model_runs = None

    # start the runs
    for x in tqdm(range(0, runs)):
        model = JointModel.from_pretrained(bert_model, intents = out_int, slots = out_slot).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None
        patience_p = patience

        for e in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, model, criterion_intents=criterion_intents, criterion_slots=criterion_slots,  clip=clip, lang=lang)
            if e % 5 == 0:
                sampled_epochs.append(e)
                losses_train.append(np.asarray(loss).mean())

                results_dev, intent_res, loss_dev = eval_loop(dev_loader, model, lang, criterion_intents=criterion_intents, criterion_slots=criterion_slots)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

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
        results_test, intent_test, _ = eval_loop(test_loader, best_model, lang, criterion_intents=criterion_intents, criterion_slots=criterion_slots)
        best_model.to("cpu")

        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

        #print(f"Intent Acc: {intent_test['accuracy']:.3f} Slot F1: {results_test['total']['f']:.3f}")
        #wandb.log({"Slot F1": results_test['total']['f'], "Intent Acc": intent_test['accuracy']})

        if results_test['total']['f'] > best_f1_runs:
            best_f1_runs = results_test['total']['f']
            #best_model_runs = deepcopy(best_model)
            #show plot
            # plt.plot(sampled_epochs, losses_train, label='Train')
            # plt.plot(sampled_epochs, losses_dev, label='Dev')
            # plt.legend()
            # plt.title(f"Loss {test_name}")
            # plt.show()

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)

    #wandb.log({"Slot F1": round(slot_f1s.mean(),3), "Intent Acc": round(intent_acc.mean(), 3)})
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
    
    # Save the model
    path = 'model_bin/JointModel.pt'
    torch.save(best_model.state_dict(), path)
    