### Usage

This script is used for an **Intent and Slot Filling Task** with the following optional flags:

- `--bid`: Adds bidirectionality to the model.
- `--drop`: Adds a dropout layer to the model.

Both flags are optional. If not provided, their corresponding functionality is **disabled** by default.

Flags can be used independently or together as needed.

#### Basic Usage (No Flags)

If you run the script without any flags, neither bidirectionality nor the dropout layer will be added.

```bash
python3 main.py
```


#### Enabling Bidirectionality

To enable bidirectionality, use the `--bid` flag:

```bash
python3 main.py --bid
```

#### Enabling Both Bidirectionality and Dropout Layer

You can also enable both features by using both flags:

```bash
python3 main.py --bid --drop
```




