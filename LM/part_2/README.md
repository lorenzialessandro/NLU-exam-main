### Usage

#### Positional Arguments:
- `optimizer`: The **optimizer** to use. Choose between:
  - `SGD`
  - `NTAvSGD`

#### Optional Flags:
- `--weight_tying`: Use **weight tying** in the model. This flag is optional, and if not provided, weight tying will be **not used** by default.
- `--var_dropout`: Use **variational dropout** in the model. This flag is optional, and if not provided, variational dropout will be **not used** by default.

### Possible combinations:

1. **LSTM with SGD optimizer, no weight tying, no variational dropout:**

   ```bash
   python3 main.py LSTM SGD
   ```

2. **LSTM with SGD optimizer, with weight tying, no variational dropout:**

   ```bash
   python3 main.py LSTM SGD --weight_tying
   ```
3. **LSTM with SGD optimizer, with weight tying and with variational dropout:**

   ```bash
   python3 main.py LSTM SGD --weight_tying --var_dropout
   ```

4. **LSTM with NTAvSGD optimizer, with weight tying and with variational dropout:**

   ```bash
   python3 main.py LSTM NTAvSGD --weight_tying --var_dropout
   ```

