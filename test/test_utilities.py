import pytest
import os
import pandas as pd
import numpy as np
import logging
from utilities import set_logger, load_data, preprocess_text, preprocess_array_of_texts, create_combined_text, extract_price

def test_set_logger(config):
    """
    Test the set_logger function.
    """
    set_logger(config)

def test_load_data(config):
    """
    Test the load_data function.
    """
    data_path = config['general'].get('data_path', 'test/resources/test_data.jsonl.gz')

    sample_size_for_training = config['tfidf'].get('sample_size_for_training', 0.01)
    data = pd.DataFrame(load_data(path=config['general'].get('data_path', '../data/amz_products_small.jsonl.gz'),
                                   step=int(1/sample_size_for_training)))

    assert data.shape[0] > 0, "Data should not be empty"

def test_preprocess_text():
    """
    Test the preprocess_text function.
    """
    text = "Hello, World! This is a test."
    stop_words = ["is", "a"]
    processed_text = preprocess_text(text, stop_words)
    assert processed_text == "hello world this test", "Text should be preprocessed correctly"

def test_preprocess_array_of_texts():
    """
    Test the preprocess_array_of_texts function.
    """
    texts = ["Hello, World!", "This is a test."]
    stop_words = ["is", "a"]
    processed_texts = preprocess_array_of_texts(texts, stop_words)
    assert processed_texts == ["hello world", "this test"], "Array of texts should be preprocessed correctly"

def test_create_combined_text():
    """
    Test the create_combined_text function.
    """
    df = pd.DataFrame({
        'col1': ["Hello", "This"],
        'col2': ["World", "is a test"]
    })
    combined_df = create_combined_text(df, ['col1', 'col2'])
    assert 'text' in combined_df.columns, "Combined DataFrame should contain 'text' column"
    assert combined_df['text'].tolist() == ["Hello World", "This is a test"], "Text should be combined correctly"

def test_extract_price():
    """
    Test the extract_price function.
    """

    df = pd.DataFrame({
        'price': ["$10.00", "$20.00-$30.00", "invalid", None]
    })
    extracted_prices_df = extract_price(df)
    expected_prices_df = pd.DataFrame({
        'price': [10.00, 25.00, np.nan, np.nan]
    })
    pd.testing.assert_frame_equal(extracted_prices_df, expected_prices_df, check_names=False)

