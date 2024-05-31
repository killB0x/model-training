"""
Tokenizing the dataset.
"""

import pickle
import pandas as pd
from lib_ml import preprocessing


def preprocess_data():
    """
    Converts csv dataset files into processed splits.
    """
    raw_train = pd.read_csv('./data/raw/train.csv', dtype='string')
    raw_val = pd.read_csv('./data/raw/val.csv', dtype='string')
    raw_test = pd.read_csv('./data/raw/test.csv', dtype='string')

    preprocessed = preprocessing.process_data(raw_train, raw_val, raw_test)

    with open('./data/processed/char_index.pkl', 'wb') as file:
        pickle.dump(preprocessed['char_index'], file)
    with open('./data/processed/tokenizer.pkl', 'wb') as file:
        pickle.dump(preprocessed['tokenizer'], file)

    with open('./data/processed/x_train.pkl', 'wb') as file:
        pickle.dump(preprocessed['x_train'], file)
    with open('./data/processed/x_val.pkl', 'wb') as file:
        pickle.dump(preprocessed['x_val'], file)
    with open('./data/processed/x_test.pkl', 'wb') as file:
        pickle.dump(preprocessed['x_test'], file)

    with open('./data/processed/y_train.pkl', 'wb') as file:
        pickle.dump(preprocessed['y_train'], file)
    with open('./data/processed/y_val.pkl', 'wb') as file:
        pickle.dump(preprocessed['y_val'], file)
    with open('./data/processed/y_test.pkl', 'wb') as file:
        pickle.dump(preprocessed['y_test'], file)

if __name__ == "__main__":
    preprocess_data()
