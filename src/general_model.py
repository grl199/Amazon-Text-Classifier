from abc import ABC, abstractmethod
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import logging

# Configure the logger
logger = logging.getLogger('general_model')

class GeneralModel(ABC):
    def __init__(self,
                 cat_cols: list = None, 
                 num_cols: list = None, 
                 label_col: str = 'main_cat',
                 model_name: str = 'prajjwal1/bert-tiny'):
        """
        Initialize the GeneralModel class.

        Args:
            cat_cols (list): Categorical columns.
            num_cols (list): Numerical columns.
            label_col (str): Label column.
            model_name (str): Model name.
        """
        if cat_cols is None:
            cat_cols = ['brand', 'description', 'feature', 'image', 'title']
        if num_cols is None:
            num_cols = ['price']

        self.label_col = label_col  
        self.cat_columns = cat_cols
        self.num_cols = num_cols
        self.model_name = model_name

        logger.info(f"Initialized GeneralModel with label_col={label_col}, model_name={model_name}")

    @abstractmethod
    def fit(self, X):
        """
        Abstract method to fit the model.

        Args:
            X (pd.DataFrame): Training data.
        """
        pass
        
    @abstractmethod
    def predict(self, X):
        """
        Abstract method to predict with the model.

        Args:
            X (pd.DataFrame): Input data for prediction.
        """
        pass

    @abstractmethod
    def save_model(self, model_path, tokenizer_path):
        """
        Abstract method to save the model.

        Args:
            model_path (str): Path to save the model.
            tokenizer_path (str): Path to save the tokenizer.
        """
        pass

    @abstractmethod
    def load_model(self, model_path, tokenizer_path):
        """
        Abstract method to load the model.

        Args:
            model_path (str): Path to load the model.
            tokenizer_path (str): Path to load the tokenizer.
        """
        pass
    
    def evaluate_model(self, sample=1, batch_size=16):
        """
        Evaluate the model on training and test sets.

        Args:
            sample (float): Proportion of data to use for evaluation.
            batch_size (int): Batch size for prediction.

        Returns:
            pd.DataFrame: DataFrame with evaluation metrics.
        """
        if not all(attr in self.__dict__ for attr in ['X_train', 'X_test', 'y_train', 'y_test']):
            raise ValueError('You must fit the model first')
        
        # Take a sample of the DataFrames X_train and y_train according to the sample parameter
        X_train = self.X_train.sample(int(sample * len(self.X_train)))
        y_train = self.y_train.loc[X_train.index]

        assert int(sample * len(self.X_test))>1, 'No rows! Increase the sample size for model evaluation!'
        X_test = self.X_test.sample(int(sample * len(self.X_test)))
        y_test = self.y_test.loc[X_test.index]

        logger.info(f"Evaluating model with sample size {sample} and batch size {batch_size}")

        # Predict on training and test sets
        y_train_pred = self.predict(X_train, batch_size=batch_size)
        y_test_pred = self.predict(X_test, batch_size=batch_size)

        if hasattr(self, 'label_mapping'):
            y_train_pred = [self.label_mapping[label] for label in y_train_pred]
            y_test_pred = [self.label_mapping[label] for label in y_test_pred]

        # Calculate metrics for the training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_report = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)

        # Calculate metrics for the test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

        logger.info(f"Train Accuracy: {train_accuracy}")
        logger.info(f"Test Accuracy: {test_accuracy}")

        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Train': [
                train_accuracy,
                train_report['weighted avg']['precision'],
                train_report['weighted avg']['recall'],
                train_report['weighted avg']['f1-score']
            ],
            'Test': [
                test_accuracy,
                test_report['weighted avg']['precision'],
                test_report['weighted avg']['recall'],
                test_report['weighted avg']['f1-score']
            ]
        })

        logger.info("Evaluation metrics calculated successfully")

        return metrics_df