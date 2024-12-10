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

## Installation

Follow these steps to get the project up and running on your local machine.

### Prerequisites:
- Python 3.x
- `pip` (Python package manager)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/false-alarm-detection.git
---

### API Endpoints**
Provide details about the API endpoints, including the request method, description, request body, and possible responses.

```markdown
## API Endpoints

### 1. Train Model
- **URL**: `/train_model`
- **Method**: `GET`
- **Description**: Trains a logistic regression model using historical alarm data and saves the trained model to a file.
- **Response**:
  - `"Model Trained Successfully"`

### 2. Test Model
- **URL**: `/test_model`
- **Method**: `POST`
- **Description**: Accepts sensor data and returns a prediction on whether the alarm is a false or true alarm.
- **Request Body (JSON)**:
  ```json
  {
    "Ambient Temperature": 7,
    "Calibration": 57.00,
    "Unwanted substance deposition": 0,
    "Humidity": 90,
    "H2S Content": 7,
    "detected by": 38
  }

---

### **Usage Instructions**
Include any extra steps for using the API or how to interact with it.

```markdown
## Usage

1. Start the Flask server:
   ```bash
   python app.py


---

### Dependencies**
List all the libraries and tools required to run the project, and include a `requirements.txt` file.

```markdown
## Dependencies

- Flask
- pandas
- scikit-learn
- numpy
- joblib
