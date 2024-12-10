# Import the necessary libraries
from flask import Flask, jsonify, request  # Flask for creating the API
import pandas as pd  # pandas for data manipulation
from sklearn.linear_model import LogisticRegression  # Logistic Regression model for classification
import numpy as np  # numpy for numerical operations
import joblib  # joblib for saving and loading the trained model

# Initialize the Flask application
app = Flask(__name__)

# Define the route for training the model
@app.route('/train_model', methods=['GET'])
def train():
    # Load the dataset from an Excel file
    data = pd.read_excel('C:/Users/Makarand/Downloads/Historical Alarm Cases.xlsx')

    # Define the independent variables (features) and dependent variable (target)
    x = data.iloc[:, 1:7]  # Select columns 1 to 7 as features (Ambient Temperature, Calibration, etc.)
    y = data['Spuriosity Index']  # Define the target column 'Spuriosity Index'

    # Create a Logistic Regression model
    lm = LogisticRegression()

    # Train the model using the provided data
    lm.fit(x, y)

    # Save the trained model to a file using joblib (this allows reusability)
    joblib.dump(lm, 'train.pkl')

    # Return a success message indicating the model was trained successfully
    return "Model Trained Successfully"

# Define the route for testing the model
@app.route('/test_model', methods=['POST'])
def test():
    # Load the trained model from the file
    pkl_file = joblib.load('train.pkl')

    # Get the input data from the POST request (JSON format)
    test_data = request.get_json()

    # Select only the relevant columns from the incoming data
    my_test_data = test_data[['Ambient Temperature','Calibration','Unwanted substance deposition','Humidity','H2S Content', 'detected by']]

    # Convert the data to a numpy array for input to the model
    my_data_array = np.array(my_test_data)

    # Reshape the array into the correct format for the model (1 sample, 6 features)
    test_array = my_data_array.reshape(1, 6)

    # Convert the reshaped array into a pandas DataFrame
    df_test = pd.DataFrame(test_array)

    # Make a prediction using the trained model
    predictions = pkl_file.predict(df_test)

    # Check the prediction and return the appropriate response
    if predictions != 1:  # If the prediction is not '1' (not dangerous)
        return "True Alarm. Danger !"  # The alarm is true and dangerous
    else:  # If the prediction is '1' (false alarm)
        return "False Alarm. No Danger."  # The alarm is false, no danger

# Run the Flask application on port 8880
app.run(port=8880)
