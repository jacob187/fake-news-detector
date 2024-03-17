import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import keras_core as keras
import keras_nlp
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import (
    Adam,
)  # Optimizer that implements the Adam algorithm.

data = pd.read_pickle("../data/processed/data.pk1")

# 512 is the limit of DistilBert
SEQ_LENGTH = 512

# Use a shorter sequence length.
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=SEQ_LENGTH,
)

# Pretrained classifier.
classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
    activation=None,
    preprocessor=preprocessor,
)

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=5e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

classifier.backbone.trainable = False

classifier.summary()
