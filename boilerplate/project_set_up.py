import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from tlm.helpers import (
    FEATURES,
)


def construct_and_save_df() -> str:
    """Create dummy data and save to a file"""
    X, y, coef = make_regression(
        n_samples=1_000_000, n_features=100, n_informative=20, coef=True
    )

    idx = np.argsort(coef)[::-1]
    columns = [f"x_{i}" for i in range(X.shape[1])]

    # give columns meaningful names
    for i, sorted_i in enumerate(idx):
        if i < 20:
            columns[sorted_i] = FEATURES[i]

    df = pd.DataFrame(X, columns=columns)

    fd, path = tempfile.mkstemp()
    df.to_parquet(path)
    return path


if __name__ == "__main__":
    data_path = construct_and_save_df()
    print(f"Wrote data to {data_path}")
