To run the file:

```python
python3 main.py <model> <optmizer>
```

The available models are:
- `LSTM_weight_tying` : LSTM that apply weight tying 
	- optimizer `SGD`
- `LSTM_VariationalDropout` : LSTM that apply variational dropout and weight tying 
	1. optimizer `SGD`
	2. optimizer `NTAvSG`

So we possible combinations are:

```python
1. python3 main.py LSTM_weight_tying SGD
2. python3 main.py LSTM_VariationalDropout SGD
3. python3 main.py LSTM_VariationalDropout NTAvSG
```

The program will automatically adapt the parameters based on the model / optimizer chosen. 
