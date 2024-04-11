import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras
import keras_nlp
from tensorflow.keras.optimizers import (
    Adam,
)


def define_model() -> keras.Model:
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

    return classifier


if __name__ == "__main__":
    classifer = define_model()
    print(classifer.summary())
