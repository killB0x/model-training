"""
Tokenizing the dataset.
"""

import pickle
import pandas as pd
from lib_ml import preprocessing


def preprocess_data():
    raw_train = pd.read_csv('./data/raw/train.csv', dtype='string')
    raw_val = pd.read_csv('./data/raw/val.csv', dtype='string')
    raw_test = pd.read_csv('./data/raw/test.csv', dtype='string')

    
    preprocessed = preprocessing.process_data(raw_train, raw_val, raw_test)

    with open('./data/processed/char_index.pkl', 'wb') as f:
        pickle.dump(preprocessed['char_index'], f)
    with open('./data/processed/tokenizer.pkl', 'wb') as f:
        pickle.dump(preprocessed['tokenizer'], f)

    with open('./data/processed/x_train.pkl', 'wb') as f:
        pickle.dump(preprocessed['x_train'], f)
    with open('./data/processed/x_val.pkl', 'wb') as f:
        pickle.dump(preprocessed['x_val'], f)
    with open('./data/processed/x_test.pkl', 'wb') as f:
        pickle.dump(preprocessed['x_test'], f)

    with open('./data/processed/y_train.pkl', 'wb') as f:
        pickle.dump(preprocessed['y_train'], f)
    with open('./data/processed/y_val.pkl', 'wb') as f:
        pickle.dump(preprocessed['y_val'], f)
    with open('./data/processed/y_test.pkl', 'wb') as f:
        pickle.dump(preprocessed['y_test'], f)

if __name__ == "__main__":
    preprocess_data()
