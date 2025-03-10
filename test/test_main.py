import pytest
import json
from src.main import app, main as main_function
from src.utilities import read_config
import argparse
from unittest.mock import patch

@pytest.fixture
def client():
    """
    Fixture to provide a test client for the Flask app.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """
    Test the home route.
    """
    response = client.get('/')
    assert response.status_code == 200, "Home route should return status code 200"
    assert b"Amazon Product Classification" in response.data, "Home page should contain 'Amazon Product Classification'"


def test_main_train(mocker):
    """
    Test the main function with the 'train' command.
    """
    config = read_config('test/resources/config_test.yaml')
    mocker.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(command='train', model='tfidf'))
    main_function(config)

@patch('sys.exit')
def test_main_api(mock_exit, mocker, client):
    """
    Test the main function with the 'api' command.
    """
    config = read_config('test/resources/config_test.yaml')
    mocker.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(command='api', model='tfidf'))
    mocker.patch('src.main.app.run')
    main_function(config)
    app.run.assert_called_once_with(host='0.0.0.0', port=4000, debug=True)

    sample_data = {
        'brand': 'BrandA',
        'description': ['This is a great product.'],
        'feature': ['Feature1'],
        'image': ['Image1'],
        'price': '$10.00',
        'title': 'Product1'
    }

    response = client.post('/predict', data=json.dumps(sample_data), content_type='application/json')
    assert response.status_code == 200, "Predict route should return status code 200"
    response_data = json.loads(response.data)
    assert 'main_cat' in response_data, "Response should contain 'main_cat' key"