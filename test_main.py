import pytest
import json
from app import app, preprocess, preprocess_data, load_model
from flask import jsonify
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Sample input JSON for testing
input_json = {
    'instant': 100,
    'dteday': '2022-12-31',
    'casual': 50,
    'registered': 100,
    'some_column': 5
}

# Test preprocess_data function
def test_preprocess_data():
    # Create a sample dataframe
    df = pd.DataFrame({
        'instant': [100],
        'dteday': ['2022-12-31'],
        'casual': [50],
        'registered': [100],
        'some_column': [5]
    })
    
    # Apply preprocessing
    result = preprocess_data(df)
    
    # Check if the result has dropped the expected columns
    assert 'instant' not in result.columns
    assert 'dteday' not in result.columns
    assert 'casual' not in result.columns
    assert 'registered' not in result.columns
    assert 'some_column' in result.columns

# Test preprocess function
def test_preprocess():
    result = preprocess(input_json)
    assert isinstance(result, pd.DataFrame)
    assert 'some_column' in result.columns

# Test prediction route with mock model
@patch('app.load_model')
@patch('app.model.predict')
def test_predict(mock_predict, mock_load_model):
    # Mock the model load and prediction
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_predict.return_value = np.array([100])
    
    # Create a Flask test client
    client = app.test_client()
    
    # Send a POST request to the '/predict' endpoint
    response = client.post('/predict', json=input_json)
    
    # Assert the response status code and the returned data
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert 'predictions' in response_json
    assert response_json['predictions'] == [100]

# Test error handling in prediction route
@patch('app.load_model')
@patch('app.model.predict')
def test_predict_error(mock_predict, mock_load_model):
    # Simulate an error during prediction
    mock_load_model.side_effect = Exception("Model loading failed")
    
    client = app.test_client()
    
    # Send a POST request to the '/predict' endpoint
    response = client.post('/predict', json=input_json)
    
    # Assert the error response
    assert response.status_code == 400
    response_json = json.loads(response.data)
    assert 'error' in response_json
    assert response_json['error'] == "Model loading failed"

# Test preprocess with invalid input
def test_preprocess_invalid_input():
    invalid_input_json = {'invalid_key': 'invalid_value'}
    
    response = preprocess(invalid_input_json)
    
    # Assert the error response
    assert isinstance(response, tuple)
    assert response[1] == 400
    assert 'error' in response[0].json