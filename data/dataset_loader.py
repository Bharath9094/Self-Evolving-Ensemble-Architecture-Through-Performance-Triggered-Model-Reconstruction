"""Local CSV and Excel dataset readers."""

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_dataset(file, filename):
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file format. Upload CSV, XLSX, or XLS.")
    try:
        if extension == ".csv":
            frame = pd.read_csv(file)
        else:
            frame = pd.read_excel(file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded dataset is empty.") from exc
    if frame.empty:
        raise ValueError("The uploaded dataset is empty.")
    frame.columns = [str(column) for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Dataset columns must have unique names.")
    return frame

