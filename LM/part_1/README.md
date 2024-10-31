# NLU course projects: Lab. 4 (LM) - part 1
This folder contains the work done to solve the **LM task** *(part 1)*, in particular to implement and improve the performance of different architectures for the **next-word prediction** task.

For this part the comparison is done, in terms of architecture, between the **RNN** and **LSTM** and, for optimizers, between **SGD** and **AdamW**. Regarding the regularization instead, **dropout** is part of the experiments. A lot of effort was also done in the hyperparameter (**learning rate**) optimization. 

Look at the report [LM.pdf](../LM.pdf) for more details. 

## Usage

### Positional Arguments:
- `architecture`: The type of **model** architecture to use. Choose between:
  - `RNN`
  - `LSTM`
  
- `optimizer`: The **optimizer** to use. Choose between:
  - `SGD`
  - `AdamW`

### Optional Flags:
- `--dropout`: Adds a **dropout** layer to the model. This flag is optional, and if not provided, dropout will be **disabled** by default.

### Basic Usage (No Dropout)

If you run the script without the `--dropout` flag, dropout will not be added:

```bash
python3 main.py RNN SGD
```

### Enabling Dropout

To enable dropout, use the `--dropout` flag:

```bash
python3 main.py LSTM SGD --dropout
```


## Possible combinations:

1. **RNN with SGD optimizer, no dropout:**

   ```bash
   python3 main.py RNN SGD
   ```

2. **LSTM with SGD optimizer, no dropout:**

   ```bash
   python3 main.py LSTM SGD
   ```
3. **LSTM with SGD optimizer, with dropout:**

   ```bash
   python3 main.py LSTM SGD --dropout
   ```

4. **LSTM with AdamW optimizer, with dropout:**

   ```bash
   python3 main.py LSTM AdamW --dropout
   ```

