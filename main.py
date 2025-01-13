import pandas as pd
from flask import Flask, request, jsonify
from category_encoders import HashingEncoder
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

app = Flask(__name__)
model = None

def preprocess_data(df):
    # df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
    df = df.fillna(df.mode().iloc[0])
    df = df.drop_duplicates()
    #Drop uncessary columns :
    col = ['instant', 'dteday', 'casual', 'registered']
    #Removing time based data and redundant data
    df.drop(col, axis = 1, inplace=True)
    return df_le

def _load_model():
    file_path = "artifacts/regression_model.pkl"
    model = pickle.load(open(file_path, "rb"))
    return model

def load_model():
    storage_client = storage.Client()
    bucket_name = "sid-ml-ops"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob("ml-artifacts/regression_model.pkl")
    blob.download_to_filename("regression_model.pkl")
    model = pickle.load(open("regression_model.pkl", "rb"))
    return model

def preprocess(input_json):
    try:
        df = pd.DataFrame(input_json, index=[0])
        x = preprocess_data(df)
        x.fillna(0, inplace=True)
        return x
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    try : 
        input_json = request.get_json()
        df_preprocessed = preprocess(input_json)
        y_predictions = model.predict(df_preprocessed)
        response = {'predictions': y_predictions.tolist()}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5051)))