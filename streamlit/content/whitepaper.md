# Fake News Detection System: A Technical Whitepaper

## Introduction

In a time characterized by information overload, the widespread dissemination of fake news undermines democracy and negatively alters our shared reality. This whitepaper explains my process in creating a robust fake news detection system.

## Problem Statement

The problem of fake news is not just about misinformation, but also about the erosion of trust in media and the potential for societal harm. My project aims to tackle this issue by creating a system that can accurately detect fake news.

## Objective

The objective of our project is to develop an artificial intelligent (AI) machine learning model that can accurately classify news articles as "true" or "fake".

## Methodology

As with many data science and AI projects, the first step in development is to analyze the data. I conducted data analysis using Excel, Jupyter Notebook, and the Python library Pandas. For visualizing the data, I used Matplotlib and Seaborn as well as Excel. I then leveraged Python’s extensive machine learning ecosystem to build my AI model.

## Dataset

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection) and is licensed under [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/). The data was presented in two CSV files: true.csv and fake.csv. Combined, there are 44,689 rows of data.

Both datasets have the following columns: text, title, subject, and date. The subjects for the true dataset are Politics News and World News, both with close to equal representation – approximately 11,000 instances vs. 10,000 instances respectively. The subjects for the fake dataset are News, Politics, Left News, Government News, US News, and Middle East. Most of the articles have the subjects News and Politics, with Left News having the third most instances. The true dataset’s dates range from January 2016 to December 2017. The dates in the fake dataset range from April 2015 to December 2017.

## Implementation

The two datasets were combined to one with the addition of the column True to label the rows as either true or false (1 or 0) as this is a binary classification model. I used the DistilBERT pretrained model and with Adam as the optimizer. The learning rate for the optimizer was set as 5e-4 and I utilized the SparseCategoricalAccuracy to evaluate the model during training.

I trained the model using the Python Library’s TensorFlow, Keras, Keras_NLP, and Sklearn libraries. The data was partitioned twice in the process. Initially, 70% of the data was set aside as a temporary training set, while the remaining 30% was allocated for testing. The temporary training set was then further divided, with approximately 67% used for actual training and the remaining 33% used as a validation set during the training process for model evaluation and tuning. I set the epoch to one which means that the entire dataset was passed through once. The provided code outlines the above descriptions on how I partitioned, validated, and tested the data on my machine:

```python
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
        x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=1, batch_size=128
    )
```

Notably, of all the information available in the datasets (text, title, subject, and date) I only used the text data in the creation of the AI model.

## Results

The validation sparse categorical accuracy was 0.9552 upon build completion. When tested on 1000 rows of the dataset, the model achieved an F1 score of 97. The model had more False Positives than False Negatives.

## Discussion

Compiling the AI model proved to be a challenging part of this project. I had initially set the epoch value to three with a batch size of 64 but my computer’s hardware could not handle this task. I decided to set the epoch to two and run the model within a GitHub Codespace. However, maintaining a constant internet connection made this approach challenging. I then created a Virtual Machine with Microsoft Azure’s cloud, but the hardware for the virtual machine was very slow – the Virtual Machine had 4vCPUs with 16GB of RAM. After changing the epoch to one and the batch size to 128 the runtime estimated the model would take 9 hours to complete within the VM. Close to the end of this process my VM was shut down due to server maintenance. I decided then to run the model on my MacBook with a 2.2 GHz 6-Core Intel Core i7 CPU which resulted in the model in which this web application utilizes.

The model performed well on the test data. Its mistakes mostly come as false positives. This was observed using a confusion matrix on 1000 rows of the training dataset as well as manual testing through ten example articles from AP News, The New York Times, Fox News, and The Onion. When I tested the model with articles from The Onion – a popular website for comedic fake news – it misclassified some of the articles as true.

## Future Work

I plan to expand the web application functionality to include database connectivity for users to store results, a web scraping feature where users can enter a URL and the text is extracted, and an automatic retraining feature for the model. I also believe that other AI tools such as sentiment analysis could help shine light into the models working. AI models are a black box and any tool I can utilize to better understand the functionality with help strength my understanding of the model and provide the user with a more informed response.

To improve the AI model there are different approaches I could take. I could increase the epoch parameter and decrease the batch size and run the code on a more powerful VM. This would likely yield a model with a higher accuracy score and lower loss rate. The training time would be longer, however, and it is unclear the degree to which the accuracy would increase. The underlying approach could also be altered. If I used the same dataset, I could train the model to take into consideration the headline and subject. This could provide the model with more contextual information, which could lead to more reliable results.

## Conclusion

My project demonstrates that it is possible to develop a machine learning model that can accurately detect fake news. While there are still challenges to overcome, the results are nevertheless promising.
