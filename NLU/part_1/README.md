# NLU course projects: Lab. 4 (NLU) - part 1

This folder contains the work done to solve the **NLU task** (*part 1*), in particular to implement and improve the performance of different architectures for the task of **Intent classification** and **Slot filling**. *Intent Classification* is a particular task of text classification, consisting in deducing (classify) the specific purpose of a whole sequence given. *Slot filling* (or *Concept tagging*) is a case of shallow parsing task that aims to identify meaningful slots in a text. 

This part consists in modifying the baseline architecture based on LSTM, adding some regularization techniques. 

Look at the report [NLU.pdf](../NLU.pdf) for more details. 

## Usage

This script is used for an **Intent and Slot Filling Task** with the following optional flags:

- `--bid`: Adds bidirectionality to the model.
- `--drop`: Adds a dropout layer to the model.

Both flags are optional. If not provided, their corresponding functionality is **disabled** by default.

Flags can be used independently or together as needed.

### Basic Usage (No Flags)

If you run the script without any flags, neither bidirectionality nor the dropout layer will be added.

```bash
python3 main.py
```


### Enabling Bidirectionality

To enable bidirectionality, use the `--bid` flag:

```bash
python3 main.py --bid
```

### Enabling Both Bidirectionality and Dropout Layer

You can also enable both features by using both flags:

```bash
python3 main.py --bid --drop
```




