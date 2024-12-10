# False-Alarm-Detection
A Flask API for predicting the severity of industrial alarms using machine learning.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)

## Project Overview

This project aims to minimize the cost of false alarms in industrial environments. Using a machine learning model trained on sensor data, the system predicts whether a gas leak detected by sensors is hazardous or not, helping companies avoid unnecessary emergency responses.

## Features
- **True vs. False Alarm Classification**: Predicts if an alarm is a true or false alarm based on sensor data.
- **Machine Learning**: Uses Logistic Regression for classification.
- **Flask API**: Provides endpoints to train and test the model via HTTP requests.

## Technologies Used
- **Python**: The primary programming language.
- **Flask**: For creating the API.
- **Scikit-learn**: For machine learning (Logistic Regression).
- **Pandas**: For data manipulation and preprocessing.
- **Numpy**: For numerical operations.
- **Joblib**: For saving and loading the trained model.
- **Postman**: For testing API endpoints.

