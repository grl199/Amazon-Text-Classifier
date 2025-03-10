import pytest
import pandas as pd
import os
from bert_classifier import BertClassifier

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
def bert_classifier():
    """
    Fixture to provide an instance of BertClassifier for testing.
    """
    return BertClassifier()

def test_fit(bert_classifier, sample_data):
    """
    Test the fit method of BertClassifier.
    """
    bert_classifier.fit(sample_data)
    assert bert_classifier.model is not None, "Model should be initialized after fitting the model"

def test_predict(bert_classifier, sample_data):
    """
    Test the predict method of BertClassifier.
    """
    bert_classifier.fit(sample_data)
    predictions = bert_classifier.predict(sample_data)
    assert len(predictions) == len(sample_data), "Number of predictions should match the number of samples"

def test_save_model(bert_classifier,sample_data, tmp_path):
    """
    Test the save_model method of BertClassifier.
    """
    model_path = tmp_path / "saved_model"
    tokenizer_path = tmp_path / "saved_tokenizer"

    bert_classifier.fit(sample_data)
    bert_classifier.save_model(model_path, tokenizer_path)
    assert os.path.exists(model_path), "Model file should be saved"
    assert os.path.exists(tokenizer_path), "Tokenizer file should be saved"

def test_load_model(bert_classifier,sample_data,tmp_path):
    """
    Test the load_model method of BertClassifier.
    """
    model_path = tmp_path / "saved_model"
    tokenizer_path = tmp_path / "saved_tokenizer"

    bert_classifier.fit(sample_data)
    bert_classifier.save_model(model_path, tokenizer_path)
    loaded_model = bert_classifier.load_model(model_path, tokenizer_path)
    assert loaded_model is not None, "Loaded model should not be None"
    assert isinstance(loaded_model, BertClassifier), "Loaded model should be an instance of BertClassifier"