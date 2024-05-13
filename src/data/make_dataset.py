"""
Loading the dataset.
"""
import pandas as pd
from lib_ml import preprocessing

DATA_FOLDER = "./data"

def make_dataset():
    for split in ['train', 'test', 'val']:
        with open(DATA_FOLDER + "/external/" + split + ".txt", "r", encoding='utf-8') as f:
            raw = preprocessing.parse_data(f)
            raw.to_csv(DATA_FOLDER + '/raw/' + split + '.csv', index=False)

if __name__ == "__main__":
    make_dataset()
