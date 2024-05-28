import os
import pytest

DATA_FOLDER = "./data"

@pytest.mark.parametrize("split", ['train', 'test', 'val'])
def test_csv_files_exist(split):
    """
    Test whether the required raw 'train.csv', 'test.csv' and 'val.csv' files exist.
    """
    file_path = os.path.join(DATA_FOLDER, "raw", f"{split}.csv")
    assert os.path.exists(file_path), f"File not found: {file_path}"
