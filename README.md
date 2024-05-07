# Fake News Detector

## Description

The Fake News Detector is a project aimed at identifying and flagging fake news articles. It utilizes machine learning algorithms to analyze the content of news articles and determine their credibility. The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection) and is licensed under [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/).

## Features

- Input a news article URL or text for analysis
- Analyze the content using natural language processing techniques
- Generate a credibility score for the article
- Flag articles as potentially fake or reliable

## Installation

1. Clone the repository: `git clone https://github.com/fake-news-detector.git`
2. Ensure that you are using the specified python version is the .python-version file
3. Install the required dependencies: `pip install -r requirements.txt`
4. Download the two csv files from [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection)
5. Create the directories ./data/raw and ./data/processed
6. Place the two csv files in ./data/raw

## Usage

### Analyzing Data

To analyze the data using the Jupyter notebook `20240315-analyze-individual-datasets.ipynb`, follow these steps:

1. Open the Jupyter notebook by running the command `jupyter notebook` in your terminal.
2. Navigate to the directory where the notebook is located: `./notebooks`.
3. Click on the `20240315-analyze-individual-datasets.ipynb` file to open it.
4. Run the notebook cell by cell to execute the code and analyze the individual datasets.

### Preprocessing Data

To create the training dataset using the Jupyter notebook `preprocess.ipynb`, follow these steps:

1. Click on the `20240315-preprocess-data.ipynb` file to open it.
2. Run the notebook cell by cell to execute the code and preprocess the data.

Note: Make sure to have the necessary dependencies installed before running the notebooks. Refer to the "Installation" section in the README for more information.

## Model training

If you wish to train AI model run `python fake_news_distilbert_model_main.py` in the `./models directory`

- Hardware limitations may prevent the model from training as the code is written.
- You can adjust the code in `./models/train_model.py` accordingly.

## Streamlit Applicaiton

Navigate to `./streamlit` and run `streamlit run Home.py` to view the streamlit application

## Contributions

Contributions to the Fake News Detector project are encouraged. If you would like to contribute, please feel free to open a pull request!
