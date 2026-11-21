from io import BytesIO, StringIO

import pandas as pd
import pytest

from data.dataset_loader import read_dataset


def test_csv_upload_reader():
    frame = read_dataset(StringIO("x,target\n1,a\n2,b\n"), "sample.csv")
    assert frame.to_dict("list") == {"x": [1, 2], "target": ["a", "b"]}


def test_excel_upload_reader():
    source = BytesIO()
    pd.DataFrame({"x": [1, 2], "target": [0, 1]}).to_excel(source, index=False)
    source.seek(0)
    frame = read_dataset(source, "sample.xlsx")
    assert frame.shape == (2, 2)


def test_upload_reader_rejects_unsupported_and_empty_files():
    with pytest.raises(ValueError, match="Unsupported"):
        read_dataset(StringIO("x\n1\n"), "sample.json")
    with pytest.raises(ValueError, match="empty"):
        read_dataset(StringIO("x,target\n"), "empty.csv")
