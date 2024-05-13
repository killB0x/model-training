"""
Tokenizing the dataset.
"""

import pickle

from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from dvclive import Live

def preprocess_data():
    raw_train = pd.read_csv('./data/raw/train.csv', dtype='string')
    raw_x_train = raw_train['url'].to_list()
    raw_y_train = raw_train['label']

    raw_val = pd.read_csv('./data/raw/val.csv', dtype='string')
    raw_x_val = raw_val['url'].to_list()
    raw_y_val = raw_val['label']

    raw_test = pd.read_csv('./data/raw/test.csv', dtype='string')
    raw_x_test = raw_test['url'].to_list()
    raw_y_test = raw_test['label']

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length=200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    with open('./data/processed/tokenizer.pkl', 'wb') as f:
        pickle.dump(char_index, f)

    with open('./data/processed/char_index.pkl', 'wb') as f:
        pickle.dump(char_index, f)

    with open('./data/processed/x_train.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('./data/processed/x_val.pkl', 'wb') as f:
        pickle.dump(x_val, f)
    with open('./data/processed/x_test.pkl', 'wb') as f:
        pickle.dump(x_test, f)

    with open('./data/processed/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('./data/processed/y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    with open('./data/processed/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

    with Live() as live:
        live.log_artifact(
            str("./data/processed/tokenizer.pkl"),
            type="tokenizer",
            name="phishing-detection-tokenizer"
        )

if __name__ == "__main__":
    preprocess_data()
