import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import datasets
import torch
import logging
import json

from utilities import create_combined_text,preprocess_image_urls
from general_model import GeneralModel

# Configure the logger
logger = logging.getLogger('default_logger')

class BertClassifier(GeneralModel):
    def __init__(self,
                 cat_cols: list = None, 
                 num_cols: list = None, 
                 label_col='main_cat',
                 model_name='prajjwal1/bert-tiny'):
        """
        Initialize the BertClassifier class.

        Args:
            cat_cols (list): Categorical columns.
            num_cols (list): Numerical columns.
            label_col (str): Label column.
            model_name (str): Model name.
        """
        # Load the pre-trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        super().__init__(cat_cols, num_cols, label_col, model_name)
        logger.info(f"Initialized BertClassifier with model_name={model_name}")

    def tokenize(self, dataset):
        """
        Tokenize the dataset.

        Args:
            dataset (datasets.Dataset): Dataset to tokenize.

        Returns:
            datasets.Dataset: Tokenized dataset.
        """
        def tokenize_function(examples, field='text'):
            return self.tokenizer(examples[field], padding="max_length", truncation=True, max_length=512)

        return dataset.map(tokenize_function, batched=True)

    def compute_metrics(self, p):
        """
        Compute evaluation metrics.

        Args:
            p (transformers.EvalPrediction): Evaluation predictions.

        Returns:
            dict: Dictionary with evaluation metrics.
        """
        preds = np.argmax(p.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
        acc = accuracy_score(p.label_ids, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def preprocess(self, X):
        """
        Preprocess the input data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            tuple: Tuple containing the training and evaluation datasets.
        """
        self.label_mapping = {label: idx for idx, label in enumerate(X[self.label_col].astype('category').cat.categories)}
        self.inv_label_mapping = {v: k for k, v in self.label_mapping.items()}

        df = create_combined_text(preprocess_image_urls(X), self.cat_columns)

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(pd.DataFrame(df['text']), df[self.label_col], test_size=0.2, random_state=42)

        # Map labels to integers
        self.y_train = self.y_train.to_frame(name=self.label_col)
        self.y_train[self.label_col] = self.y_train[self.label_col].map(self.label_mapping)

        self.y_test = self.y_test.to_frame(name=self.label_col)
        self.y_test[self.label_col] = self.y_test[self.label_col].map(self.label_mapping)

        self.y_train.rename(columns={self.label_col: 'labels'}, inplace=True)
        self.y_test.rename(columns={self.label_col: 'labels'}, inplace=True)
    
        train_dataset = datasets.Dataset.from_pandas(pd.concat([self.X_train, self.y_train], axis=1))
        eval_dataset = datasets.Dataset.from_pandas(pd.concat([self.X_test, self.y_test], axis=1))

        logger.info("Preprocessed the data and created training and evaluation datasets")

        return train_dataset, eval_dataset

    def fit(self, X, 
            evaluation_strategy="epoch",
            label_names=['labels'],
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=10,
            eval_steps=10,
            weight_decay=0.001):
        """
        Fit the model.

        Args:
            X (pd.DataFrame): Training data.
            evaluation_strategy (str): Evaluation strategy.
            label_names (list): List of label names.
            learning_rate (float): Learning rate.
            per_device_train_batch_size (int): Batch size for training.
            per_device_eval_batch_size (int): Batch size for evaluation.
            num_train_epochs (int): Number of training epochs.
            eval_steps (int): Number of steps between evaluations.
            weight_decay (float): Weight decay for regularization.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(X[self.label_col].unique()))
        self.model.to(self.device)

        # Create label mappings
        train_dataset, eval_dataset = self.preprocess(X)
        tokenized_train_dataset = self.tokenize(train_dataset)
        tokenized_eval_dataset = self.tokenize(eval_dataset)

        self.training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy=evaluation_strategy,
            label_names=label_names,
            learning_rate=learning_rate,
            per_device_train_batch_size=int(per_device_train_batch_size),
            per_device_eval_batch_size=int(per_device_eval_batch_size),
            num_train_epochs=int(num_train_epochs),
            eval_steps=int(eval_steps),
            weight_decay=weight_decay,
        )

        # Create the trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=self.compute_metrics
        )

        logger.info("Starting model training")
        # Train the model
        self.trainer.train()

        logger.info("Evaluating the model")
        # Evaluate the model
        self.trainer.evaluate()

        return self
    
    def predict(self, X, batch_size=32):
        """
        Predict using the model.

        Args:
            X (pd.DataFrame): Input data for prediction.
            batch_size (int): Batch size for prediction.

        Returns:
            list: List of predicted labels.
        """
        if 'text' not in X.columns:
            X = create_combined_text(X, self.cat_columns)

        # Tokenize the texts
        tokenized_texts = self.tokenizer(X['text'].tolist(), return_tensors="pt", max_length=512, padding=True, truncation=True)

        if 'token_type_ids' in tokenized_texts:
            del tokenized_texts['token_type_ids']

        # Move tensors to the device (the model is already on the device)
        tokenized_texts = {k: v.to(self.device) for k, v in tokenized_texts.items()}

        predicted_labels = []
        for i in range(0, len(X), batch_size):
            batch = {k: v[i:i + batch_size] for k, v in tokenized_texts.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            batch_predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predicted_labels.extend(batch_predicted_labels)

        logger.info("Predicted labels for the input data")

        return [self.inv_label_mapping[int(label)] for label in predicted_labels]

    def save_model(self, model_path, tokenizer_path):
        """
        Save the model and tokenizer.

        Args:
            model_path (str): Path to save the model.
            tokenizer_path (str): Path to save the tokenizer.
        """
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)

        # Read the existing config.json file
        config_path = f"{model_path}/config.json"
        with open(config_path, 'r') as file:
            config = json.load(file)

        # Add the label mappings to the config.json file
        config['label_mapping'] = self.label_mapping

        # Save the updated config.json file
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=2)

        logger.info(f"Model and tokenizer saved to {model_path} and {tokenizer_path}")

    def load_model(self, model_path, tokenizer_path):
        """
        Load the model and tokenizer.

        Args:
            model_path (str): Path to load the model.
            tokenizer_path (str): Path to load the tokenizer.

        Returns:
            BertClassifier: The loaded model.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.to(self.device)

        with open(f"{model_path}/config.json", 'r') as file:
            config = json.load(file)
            self.label_mapping = config.get("label_mapping", [])
            self.inv_label_mapping = {int(v): k for k, v in self.label_mapping.items()}

        logger.info(f"Model and tokenizer loaded from {model_path} and {tokenizer_path}")

        return self