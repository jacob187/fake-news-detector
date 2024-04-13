import create_training_dataset
import define_model
import train_model


data = create_training_dataset.load_and_shuffle_data("../data/processed/data.pk1")
classifier = define_model.define_model()


X_temp, X_test, y_temp, y_test = train_model.split_data(data, "text", "true")

train_model.fit_data(classifier, X_temp, y_temp, X_test, y_test)

classifier.save("./builds/fake-news_distilbert_model.keras")
