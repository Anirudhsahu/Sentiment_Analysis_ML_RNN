## Sentiment Analysis using Machine Learning & RNN
This repository demonstrates a sentiment analysis pipeline using both classical Machine Learning (ML) models and a Recurrent Neural Network (RNN). The notebook walks through data preprocessing, feature engineering, model building, training, and evaluation—offering a comparison between traditional ML methods and deep learning approaches for classifying text sentiment.

## Table of Contents
Overview

Project Structure

Dataset

Installation & Requirements

Usage

Models

Results

License

Contact

## Overview
Sentiment analysis classifies text as expressing positive, negative, or neutral sentiment. This project includes:

Text preprocessing (tokenization, cleaning, etc.)

Classical ML models like Logistic Regression, Naive Bayes, SVM, etc.

An RNN model (e.g., LSTM or GRU) using Keras/TensorFlow for deep learning-based sentiment analysis.

Model evaluation with metrics such as accuracy, confusion matrix, and more.

## Dataset
Data Source: The notebook uses a sentiment analysis dataset (e.g., Twitter data, IMDB reviews, or any labeled sentiment dataset).

Data Format: Typically CSV or TSV with text and label columns.

Preprocessing: Tokenization, removal of stopwords, and handling punctuation are covered in the notebook.

You can replace or supplement your own dataset, but ensure you update the file paths and preprocessing steps accordingly.

## Installation & Requirements
Clone this repository:

git clone https://github.com/Anirudhsahu/Sentiment_Analysis_ML_RNN.git

cd Sentiment_Analysis_ML_RNN

## Create and activate a virtual environment (recommended):

python -m venv venv

source venv/bin/activate       # On macOS/Linux

# or:

.\venv\Scripts\activate        # On Windows

Install required libraries:

pip install -r requirements.txt

If requirements.txt is missing, check the top of sentiment_analysis_ml_rnn.ipynb for libraries like:

numpy, pandas, matplotlib, seaborn

scikit-learn

nltk

tensorflow or keras

re (standard library, no install needed)

## Download necessary NLTK data (if the notebook requires stopwords, wordnet, etc.):

import nltk
nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

(You can run these in a Python shell or within the notebook.)

## Usage
Launch Jupyter Notebook (or any Jupyter-compatible environment, like VS Code):

jupyter notebook

Open sentiment_analysis_ml_rnn.ipynb in your web browser.

Run the cells sequentially:

Data Loading: Loads the sentiment dataset.

Data Cleaning & Preprocessing: Tokenization, stopword removal, etc.

Feature Extraction: For classical ML methods (e.g., TF-IDF).

Classical ML Models: Logistic Regression, Naive Bayes, SVM.

RNN Model: LSTM or GRU implementation using Keras/TensorFlow.

Evaluation: Check accuracy, confusion matrix, precision, recall, F1-score.

## Models
# Classical Machine Learning
Logistic Regression

Naive Bayes

SVM

Each uses TF-IDF or a similar vectorization technique, then a standard classification algorithm.

# Recurrent Neural Network (RNN)
Embedding Layer (learned or pre-trained embeddings)

LSTM/GRU Layer for sequence modeling

Dense Output Layer for final classification

## Results
Accuracy: Compare how well classical models stack up against the RNN.

Loss & Accuracy Plots: For RNN training, visualize training vs. validation.

Confusion Matrix: Inspect misclassifications and overall performance.

Exact metrics will vary depending on the dataset size, hyperparameters, and number of epochs you train for.

## Contact
Author: Anirudhsahu

Project Link: GitHub - Sentiment_Analysis_ML_RNN

For questions or feedback, feel free to open an issue or reach out via GitHub.
