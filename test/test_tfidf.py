import pytest
import pandas as pd
import os
from tfidf import TextClassifier

@pytest.fixture(scope='module')
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    data = {
        'brand': ['BrandA', 'BrandB'],
        'description': ['This is a great product.', 'This product is not good.'],
        'feature': ['Feature1', 'Feature2'],
        'image': ['Image1', 'Image2'],
        'url': ['http://example.com/product1', 'http://example.com/product2'],
        'price': ['$10.00', '$20.00'],
        'title': ['Product1', 'Product2'],
        'main_cat': ['Category1', 'Category2']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope='module')
def text_classifier():
    """
    Fixture to provide an instance of TextClassifier for testing.
    """
    return TextClassifier()

def test_fit(text_classifier, sample_data):
    """
    Test the fit method of TextClassifier.
    """
    text_classifier.fit(sample_data)
    assert text_classifier.pipeline is not None, "Pipeline should be initialized after fitting the model"

def test_predict(text_classifier, sample_data):
    """
    Test the predict method of TextClassifier.
    """
    text_classifier.fit(sample_data)
    predictions = text_classifier.predict(sample_data)
    assert len(predictions) == len(sample_data), "Number of predictions should match the number of samples"

def test_save_model(text_classifier, tmp_path):
    """
    Test the save_model method of TextClassifier.
    """
    model_path = tmp_path / "saved_model.pkl"
    text_classifier.save_model(model_path)
    assert os.path.exists(model_path), "Model file should be saved"

def test_load_model(text_classifier,tmp_path):
    """
    Test the load_model method of TextClassifier.
    """
    model_path = tmp_path / "saved_model.pkl"
    text_classifier.save_model(model_path)
    loaded_model = text_classifier.load_model(model_path)
    assert loaded_model is not None, "Loaded model should not be None"
    assert isinstance(loaded_model, TextClassifier), "Loaded model should be an instance of TextClassifier"