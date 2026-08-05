# SentimentSense: IMDb Movie Review Classifier 🎬

A complete, beginner-friendly Natural Language Processing (NLP) project to classify movie reviews using machine learning.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge)
![TF--IDF](https://img.shields.io/badge/TF--IDF-0B7261?style=for-the-badge)
![Sentiment%20Analysis](https://img.shields.io/badge/Sentiment%20Analysis-8E44AD?style=for-the-badge)

## 📌 Project Overview
SentimentSense uses the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) to demonstrate a full data science pipeline:
1.  **Preprocessing**: Cleaning HTML, stopwords, and lemmatization.
2.  **EDA**: Visualizing word distributions with WordClouds and histograms.
3.  **Modeling**: Comparing Naive Bayes (Baseline) with Logistic Regression.
4.  **Evaluation**: Using Confusion Matrices and ROC-AUC curves.

## 🚀 Setup Instructions

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/thompgt/IMDb_Sentiment_Analyzer.git
    cd IMDb_Sentiment_Analyzer
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebook**:
    Open `notebooks/sentiment_analysis.ipynb` in Jupyter or VS Code.

## 📊 Example Outputs
The project generates:
- **WordClouds** highlighting top positive/negative words.
- **Classification Report** with ~88% accuracy on the test set.
- **ROC Curve** demonstrating model performance.

## 📜 License
MIT
