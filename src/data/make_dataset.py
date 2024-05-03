"""
Loading the dataset.
"""
import pandas as pd

DATA_FOLDER = "./data"

def make_dataset():
    for split in ['train', 'test', 'val']:
        with open(DATA_FOLDER + "/external/" + split + ".txt", "r", encoding='utf-8') as f:
            data = [line.strip() for line in f.readlines()[1:]]
            raw_x = [line.split("\t")[1] for line in data]
            raw_y = [line.split("\t")[0] for line in data]
            raw = pd.DataFrame({"url": raw_x, "label": raw_y })
            raw.to_csv(DATA_FOLDER + '/raw/' + split + '.csv', index=False)

if __name__ == "__main__":
    make_dataset()
