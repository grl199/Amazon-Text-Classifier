from flask import Flask, request, jsonify, render_template
import pandas as pd
import argparse
import logging
import warnings
import os

from tfidf import TextClassifier
from bert_classifier import BertClassifier
from utilities import load_data, read_config, set_logger

# Disable deprecation warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Configure the logger
logger = logging.getLogger('default_logger')
global loaded_model


@app.route('/', methods=['GET'])
def home():
    """
    Render the home page with the Amazon logo.
    """
    with open('img/Amazon_logo.txt', 'r') as txt_file:
        image = txt_file.read()
    return render_template('front.html', image=image)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the main category of the product based on the input data.
    """
    data = request.json
    df = pd.DataFrame([data])
    prediction = loaded_model.predict(df)
    return jsonify({'main_cat': prediction[0]})

def main(config):
    """
    Main function to handle training and API commands.

    Args:
        config (dict): Configuration dictionary.
    """
    parser = argparse.ArgumentParser(description='Product Classification API')
    parser.add_argument('--command', choices=['train', 'api'], required=True, help='"train" to train the model, "api" to launch the API')
    parser.add_argument('--model', choices=['llm', 'tfidf'], required=True, help='"llm" or "tfidf"')
    args = parser.parse_args()

    # Initialize the model based on the provided argument
    if args.model == 'llm':
        model = BertClassifier(**config[args.model]['init_args'])
    else:
        model = TextClassifier(**config[args.model]['init_args'])

    if args.command == 'train':
        # Load data
        sample_size_for_training = config[args.model].get('sample_size_for_training', 0.01)
        data_path = config['general'].get('data_path', '../data/amz_products_small.jsonl.gz')
        logger.info(f'Reading dataframe from {data_path} with sample size {sample_size_for_training}')
        X = pd.DataFrame(load_data(path=data_path,
                                   step=int(1/sample_size_for_training)))

        # Train the model
        training_args = config[args.model].get('training_args', [])
        logger.info(f'Training args: {training_args}')
        model.fit(X=X, **training_args)

        # Evaluate the model
        logger.info('Results of model evaluation:')
        sample_size = config[args.model].get('sample_size_for_model_evaluation', 1.0)
        logger.info(model.evaluate_model(sample=sample_size,
                                         batch_size=int(config[args.model].get('batch_size_for_model_evaluation', 16))))

        # Save the model if specified in the configuration
        if config['general'].get('save_model', False):
            model.save_model(model_path=config[args.model]['model_path'],
                             tokenizer_path=config[args.model]['tokenizer_path'])

        logger.info('Success!')

    if args.command == 'api':
        # Load the model for API usage
        global loaded_model
        loaded_model = model.load_model(
            model_path=config[args.model]['model_path'],
            tokenizer_path=config[args.model]['tokenizer_path'])

        # Run the Flask app
        app.run(host='0.0.0.0', port=4000, debug=True)

if __name__ == '__main__':
    # Read configuration and set up logger
    #Read, if possible, config.json as environment variable (it can be modified from the outside of the container).
    #Otherwise, read the .json copied in the container
    CONFIG_PATH = os.getenv('CONFIG_PATH', 'scripts/config.yaml')  # Ruta por defecto en el contenedor

    config = read_config(CONFIG_PATH)
    logger = set_logger(config=config)

    # Run the main function
    main(config)

# Example usage:
# http://localhost:4000


