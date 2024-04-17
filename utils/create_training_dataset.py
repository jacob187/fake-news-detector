import pandas as pd
import os


def load_and_shuffle_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")

    data = pd.read_pickle(filepath)
    shuffled_data = data.sample(frac=1)
    return shuffled_data


if __name__ == "__main__":
    load_and_shuffle_data("../data/processed/data.pk1")
