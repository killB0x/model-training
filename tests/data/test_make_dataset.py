import os
import pytest
import pandas as pd

DATA_FOLDER = "./data"

@pytest.mark.parametrize("split", ['train', 'test', 'val'])
def test_txt_files_exist(split):
    """
    Test whether the required 'train.txt', 'test.txt' and 'val.txt' files exist.
    """
    file_path = os.path.join(DATA_FOLDER, "external", f"{split}.txt")
    assert os.path.exists(file_path), f"File not found: {file_path}"

@pytest.fixture
def x_df(split):
    x_df = pd.read_csv(os.path.join(DATA_FOLDER, 'raw', f'{split}.csv'), dtype='string')
    yield x_df

@pytest.mark.parametrize("split", ['train', 'test', 'val'])
def test_no_duplicates(split, x_df):
    """
    Test whether there are no duplicate urls in the dataset.
    """
    assert len(x_df['url'].unique()) == x_df.shape[0]
    assert x_df.groupby(['label','url']).size().max()==1
