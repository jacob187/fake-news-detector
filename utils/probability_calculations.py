import numpy as np


def confidence_score(predictions):
    return np.max(softmax(predictions), axis=1)


def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
