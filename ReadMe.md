# Movie Genre Prediction

This project is a machine learning-based application that predicts the genre of a movie based on its plot summary. The project is divided into two parts: model building in a Jupyter notebook and creating a user interface with Streamlit.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Building](#model-building)
- [Streamlit App](#streamlit-app)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)

## Project Overview

The goal of this project is to build a model that can classify movies into genres based on their descriptions. The model uses TF-IDF for feature extraction and classifiers like Naive Bayes, Logistic Regression, and Support Vector Machines (SVM) for prediction. The best model is then saved and used in a Streamlit app to provide predictions based on user input.

## Dataset

The dataset used for this project contains movie descriptions and their corresponding genres. The dataset was provided in a text format and was preprocessed before training the model.Create folder named Genre Classification Dataset,, Use dataset from link below:
https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## Model Building

The model building process is carried out in a Jupyter notebook. The steps include:

1. **Data Loading**: The dataset is loaded and preprocessed.
2. **Feature Extraction**: TF-IDF is used to convert movie descriptions into numerical features.
3. **Model Training**: Multiple models are trained, including Naive Bayes, Logistic Regression, and SVM.
4. **Model Evaluation**: The models are evaluated, and the best one is saved for deployment.

The trained model and TF-IDF vectorizer are saved using `joblib`.

## Streamlit App

The Streamlit app provides a simple user interface where users can input a movie description and get a predicted genre. The app loads the saved model and vectorizer to make predictions.

## How to Run

### 1. Clone the Repository

```bash
git clone <https://github.com/Anikesh20/Movie-Genre-Classification>
cd movie-genre-prediction


### 2. Install Dependencies 
   pip install -r requirements.txt
   
### 3. Run the Jupyter Notebook
    jupyter notebook
    
### 4. Run the app
   streamlit run app.py
    
   

