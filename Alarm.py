# load all the required libraries
from flask import Flask,jsonify,request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

app = Flask(__name__)


@app.route('/train_model', methods=['GET'])
def train():
    data = pd.read_excel('C:/Users/Makarand/Downloads/Historical Alarm Cases.xlsx')
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index']
    lm = LogisticRegression()
    lm.fit(x, y)
    joblib.dump(lm, 'train.pkl')
    return "Model Trained Successfully"


@app.route('/test_model', methods=['POST'])
def test():
    pkl_file = joblib.load('train.pkl')
    test_data = request.get_json()
    my_test_data = test_data[['Ambient Temperature','Calibration','Unwanted substance deposition','Humidity','H2S Content', 'detected by']]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1,6)
    df_test = pd.DataFrame(test_array)
    predictions = pkl_file.predict(df_test)

    if predictions != 1:
        return "True Alarm. Danger !"
    else:
        return "False Alarm. No Danger."


app.run(port=8880)