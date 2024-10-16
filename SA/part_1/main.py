# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

from functions import * # Import everything from functions.py file
from utils import load_data

# define parameters
bert_model = 'bert-base-uncased'
lr = 0.0001
runs = 5
n_epochs = 100
clip = 5
patience = 5
device = 'cuda:0'


def main():
    # Load the datasets
    tmp_train_raw = load_data(os.path.join('dataset','SemEval2014','laptop14_train.txt'))
    test_raw = load_data(os.path.join('dataset','SemEval2014','laptop14_test.txt'))

    # Lunch the run(s) with the parameters
    run(tmp_train_raw, test_raw, bert_model=bert_model, lr=lr, runs=runs, n_epochs=n_epochs, clip=clip, patience=patience, device=device)


if __name__ == "__main__":
    #Write the code to load the datasets and to run your functions
    # Print the results
    main()
