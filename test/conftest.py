'''
conftest.py
'''

import os
import pytest

from bert_classifier import BertClassifier
from tfidf import TextClassifier
from utilities import read_config,set_logger,load_data

CONFIG = read_config('test/resources/config_test.yaml')


@pytest.fixture(scope='session', autouse=True)
def config():
    """
    Fixture to get the configuration file for the test session.
    """
    return CONFIG

@pytest.fixture(scope='session', autouse=True)
def logger():
    """
    Fixture to set up the logger for the test session.
    """
    logger_object = set_logger(CONFIG)
    return logger_object

