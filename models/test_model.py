import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

sys.path.append("../")
from utils.clean_text import clean_text
from utils.create_training_dataset import load_and_shuffle_data
from utils.probability_calculations import confidence_score


def display_confusion_matrix(
    true_labels: pd.Series, predicted_class: np.ndarray, dataset: str = "Test"
) -> None:
    """
    Display the confusion matrix along with the F1 score.

    Parameters:
    - true_labels: pd.Series of true labels
    - predicted_class: np.ndarray of predicted labels
    - dataset: str, name of the dataset being evaluated
    """
    cm = confusion_matrix(true_labels, predicted_class)
    f1 = f1_score(true_labels, predicted_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False", "True"])
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(
        f"Confusion Matrix on {dataset} Dataset -- F1 Score: {f1.round(2)}"
    )
    plt.show()


def test_model(
    model, data_frame: pd.DataFrame, text_column: str, label_column: str
) -> None:
    """
    Test the model and display the confusion matrix.

    Parameters:
    - model: Trained Keras model
    - data_frame: pd.DataFrame containing text data and labels
    - text_column: str, name of the text column in the DataFrame
    - label_column: str, name of the label column in the DataFrame
    """
    # Preprocess text data if necessary
    data_frame[text_column] = data_frame[text_column].apply(clean_text)

    predictions = model.predict(data_frame[text_column])
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)

    confidence_scores = confidence_score(predictions)
    print(confidence_scores)

    true_labels = data_frame[label_column]
    display_confusion_matrix(true_labels, predicted_class)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try:
        model = load_model("./builds/fake-news_distilbert_model.keras")
    except FileNotFoundError as e:
        print("Model not found. Please train the model first.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    try:
        data = load_and_shuffle_data("../data/processed/data.pk1")
    except FileNotFoundError as e:
        print("Data file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit(1)

    test_model(model, data.head(100), "text", "true")

    try:
        manuel_data = pd.read_csv("../data/raw/manueltest.csv")
    except FileNotFoundError as e:
        print("Manual data file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the manual data: {e}")
        sys.exit(1)

    text_column = "text"
    manuel_data[text_column] = manuel_data[text_column].apply(lambda x: clean_text(x))
    print(manuel_data.head(10))
    test_model(model, manuel_data, text_column, "TRUE")
