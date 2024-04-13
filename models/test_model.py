import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from clean_text import clean_text
from create_training_dataset import load_and_shuffle_data


def display_confusion_matrix(
    true_labels: pd.Series, predicted_class: np.ndarray, dataset: str = "Test"
) -> None:

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_class)

    # Compute the F1 score
    f1 = f1_score(true_labels, predicted_class)

    # Create a ConfusionMatrixDisplay instance
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False", "True"])

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(
        f"Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1.round(2))
    )

    plt.show()


def test_model(model, data_frame: pd.DataFrame, text: str, labels: str):

    predictions = model.predict(data_frame[text])

    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)

    confidence_scores = confidence_score(predictions)
    print(confidence_scores, type(confidence_scores))

    true_labels = data_frame[labels]

    display_confusion_matrix(true_labels, predicted_class)


def confidence_score(predictions):
    print(type(predictions))
    return np.max(softmax(predictions), axis=1)


def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model = load_model("./builds/fake-news_distilbert_model.keras")

    data = load_and_shuffle_data("../data/processed/data.pk1")
    test_model(model, data.head(100), "text", "true")

    manuel_data = pd.read_csv("../data/raw/manueltest.csv")
    text = "text"
    manuel_data[text] = manuel_data[text].apply(lambda x: clean_text(x))
    print(manuel_data.head(10))
    test_model(model, manuel_data, text, "TRUE")
