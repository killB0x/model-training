"""
Loading the dataset.
"""
from lib_ml import preprocessing


DATA_FOLDER = "./data"

def make_dataset():
    """
    Converts raw .txt dataset files into .csv files.
    """
    for split in ['train', 'test', 'val']:
        with open(DATA_FOLDER + "/external/" + split + ".txt", "r", encoding='utf-8') as file:
            raw = preprocessing.parse_data(file)

            raw.drop_duplicates("url", inplace=True)

            raw.to_csv(DATA_FOLDER + '/raw/' + split + '.csv', index=False)

if __name__ == "__main__":
    make_dataset()
