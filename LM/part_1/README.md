### Usage

#### Positional Arguments:
- `architecture`: The type of **model** architecture to use. Choose between:
  - `RNN`
  - `LSTM`
  
- `optimizer`: The **optimizer** to use. Choose between:
  - `SGD`
  - `AdamW`

#### Optional Flags:
- `--dropout`: Adds a **dropout** layer to the model. This flag is optional, and if not provided, dropout will be **disabled** by default.

#### Basic Usage (No Dropout)

If you run the script without the `--dropout` flag, dropout will not be added:

```bash
python3 main.py RNN SGD
```

#### Enabling Dropout

To enable dropout, use the `--dropout` flag:

```bash
python3 main.py LSTM SGD --dropout
```


### Possible combinations:

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

