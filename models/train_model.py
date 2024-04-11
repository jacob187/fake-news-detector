import keras_core as keras
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, text: str, labels: str) -> tuple:
    x = data[text]
    y = data[labels]
    X_temp, X_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.33, random_state=42
    )

    return X_train, X_test, y_train, y_test


def fit_data(
    classifier: keras.Model,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> None:

    classifier.fit(
        x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64
    )


if __name__ == "__main__":
    data = pd.read_csv("../data/processed/data.csv")
    X_train, X_test, y_train, y_test = split_data(data, "text", "true")
