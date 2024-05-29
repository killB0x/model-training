import os
import pytest

DATA_FOLDER = "./data"

@pytest.mark.parametrize("split", ['train', 'test', 'val'])
def test_txt_files_exist(split):
    """
    Test whether the required 'train.txt', 'test.txt' and 'val.txt' files exist.
    """
    file_path = os.path.join(DATA_FOLDER, "external", f"{split}.txt")
    assert os.path.exists(file_path), f"File not found: {file_path}"
