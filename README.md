# YuvaIntern
# Heart Disease Prediction Using Python

## Project Overview

This project uses machine learning to predict whether a patient record shows signs of heart disease.

The project is created for learning and research purposes. It is not a medical diagnosis tool and should not replace a doctor or medical examination.

## Objectives

The main objectives of this project are to:

- Study the heart disease dataset.
- Clean and prepare the data.
- Build machine learning models in Python.
- Compare logistic regression and random forest models.
- Measure model performance.
- Create charts to understand the results.
- Document the limitations and risks of the models.

## Dataset

This project uses the Cleveland subset of the UCI Heart Disease dataset.

The dataset contains:

- 303 patient records
- 13 input features
- A target column showing whether heart disease is present

The target value is changed into two classes:

- `0` = No heart disease
- `1` = Heart disease present

Source:

[UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease )

## Features

Some of the features used in the project are:

- Age
- Sex
- Chest-pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG result
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia result

## Machine Learning Models

The project compares two models:

1. Logistic Regression
2. Random Forest

Logistic regression is used as a simple and easy-to-understand baseline model. Random forest is used as a second model for comparison.

## Data Preparation

The following steps are used:

1. Load the dataset.
2. Replace missing values.
3. Convert the original target into two classes.
4. Split the data into training and testing sets.
5. Fill missing values using the median.
6. Scale the input values where required.
7. Train the machine learning models.

The data is split into:

- 80% training data
- 20% testing data

A fixed random seed is used so that the results can be reproduced.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix

Recall is especially important because it shows how many patients with heart disease were correctly identified.

## Example Results

The example results from the test data are:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.869 | 0.813 | 0.929 | 0.867 | 0.950 |
| Random Forest | 0.885 | 0.839 | 0.929 | 0.881 | 0.963 |

These results are only benchmark results from a small historical dataset. They should not be treated as medical performance results.

## Visualizations

The project includes the following charts:

- Target distribution
- Age distribution by target
- ROC curves
- Confusion matrices
- Logistic regression feature coefficients


