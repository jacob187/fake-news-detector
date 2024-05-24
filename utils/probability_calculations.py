import numpy as np


def confidence_score(predictions: np.ndarray) -> np.ndarray:
    return np.max(softmax(predictions), axis=1)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
