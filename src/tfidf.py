import pandas as pd
import numpy as np
from abc import abstractmethod
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import ssl
import joblib
import functools

from general_model import GeneralModel
from utilities import preprocess_image_urls, extract_price, create_combined_text, extract_url_keywords

# Configure the logger
logger = logging.getLogger('tfidf_logger')

class TextClassifier(GeneralModel):
    def __init__(self,
                 cat_cols: list = None, 
                 num_cols: list = None, 
                 label_col='main_cat',
                 model_name='tfidf'):
        """
        Initialize the TextClassifier class.

        Args:
            cat_cols (list): Categorical columns.
            num_cols (list): Numerical columns.
            label_col (str): Label column.
            model_name (str): Model name.
        """
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')
        nltk.download('stopwords')

        super().__init__(cat_cols, num_cols, label_col, model_name)
        logger.info(f"Initialized TextClassifier with model_name={model_name}")

        self.pipeline = self.create_pipeline()


    def create_pipeline(self, max_features=5000, stop_words='english'):
        """
        Create the preprocessing and classification pipeline.

        Args:
            max_features (int): Maximum number of features for TfidfVectorizer.
            stop_words (str): Stop words to use in TfidfVectorizer.

        Returns:
            Pipeline: The created pipeline.
        """

        # Transformador para texto
        text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=max_features, 
                                    tokenizer=word_tokenize, 
                                    stop_words=stop_words,
                                    min_df=1))
        ])
        
        # Transformador para columnas numéricas
        num_transformer = Pipeline(steps=[
            ('extract_num', FunctionTransformer(extract_price)),
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])
        
        # Construir el `ColumnTransformer` con todas las transformaciones necesarias
        transformers = [
            ('text', text_transformer, 'text')
        ]
        
        # Agregar transformador para columnas numéricas si existen
        if self.num_cols:
            transformers.append(('num', num_transformer, self.num_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers)

        logger.info("Created preprocessing and classification pipeline")
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('clf', MultinomialNB())
        ])
    def preprocess(self, X):
        """
        Preprocess the data.

        Args:
            X (pd.DataFrame): Data to preprocess (both features and target).

        Returns:
            tuple: Tuple containing preprocessed training and test data, and their labels.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, X[self.label_col], test_size=0.2, random_state=42)
        
        # Combinar texto de columnas categóricas, prestando especial atención al campo 'image'
        # 1. Procesar URLs en 'image'
        if 'image' in self.cat_columns and 'image' in self.X_train.columns:
            self.X_train['image'] = self.X_train['image'].fillna('').astype(str)
            self.X_test['image'] = self.X_test['image'].fillna('').astype(str)
            
            # Extraer palabras clave de URLs
            self.X_train['image_keywords'] = self.X_train['image'].apply(extract_url_keywords)
            self.X_test['image_keywords'] = self.X_test['image'].apply(extract_url_keywords)
            
            # Añadir las palabras clave al texto combinado
            cat_cols = [col for col in self.cat_columns if col != 'image'] + ['image_keywords']
        else:
            cat_cols = self.cat_columns
        
        # 2. Combinar texto de todas las columnas categóricas
        X_train_merged = create_combined_text(self.X_train, cat_cols)[['text'] + self.num_cols]
        X_test_merged = create_combined_text(self.X_test, cat_cols)[['text'] + self.num_cols]
        
        logger.info("Preprocessed the data and split into training and test sets")

        return X_train_merged, X_test_merged, self.y_train, self.y_test

    def fit(self, X, max_features=5000, tokenizer=word_tokenize, stop_words='english'):
        """
        Train the model.

        Args:
            X (pd.DataFrame): Data to preprocess (both features and target).
            max_features (int): Maximum number of features for TfidfVectorizer.
            tokenizer (callable): Tokenizer to use in TfidfVectorizer.
            stop_words (str): Stop words to use in TfidfVectorizer.

        Returns:
            TextClassifier: The trained model.
        """
        X, _, y, _ = self.preprocess(X)

        self.pipeline.set_params(preprocessor__text__tfidf__tokenizer=tokenizer, 
                                 preprocessor__text__tfidf__stop_words=stop_words,
                                 preprocessor__text__tfidf__max_features=int(max_features))

        logger.info("Starting model training")
        self.pipeline.fit(X, y)
        logger.info("Model training completed")

        return self

    def predict(self, X: pd.DataFrame, batch_size=None):
        """
        Predict using the model.

        Args:
            X (pd.DataFrame): Input data for prediction.
            batch_size (int, optional): Batch size for prediction. Defaults to None.

        Returns:
            np.ndarray: Predicted labels.
        """
        logger.info("Predicting labels for the input data")
        
        # Preprocesar de manera consistente con el método fit
        X_copy = X.copy()
        
        # Procesar URLs en 'image' si existe
        if 'image' in self.cat_columns and 'image' in X_copy.columns:
            X_copy['image'] = X_copy['image'].fillna('').astype(str)
            X_copy['image_keywords'] = X_copy['image'].apply(extract_url_keywords)
            cat_cols = [col for col in self.cat_columns if col != 'image'] + ['image_keywords']
        else:
            cat_cols = self.cat_columns
        
        # Combinar texto de todas las columnas categóricas
        preprocessed_X = create_combined_text(X_copy, cat_cols)[['text'] + self.num_cols]
        
        return self.pipeline.predict(preprocessed_X)

    def save_model(self, model_path: str, tokenizer_path: str = None):
        """
        Save the model as a pickle.

        Args:
            model_path (str): Path to save the model.
            tokenizer_path (str, optional): Path to save the tokenizer. Defaults to None.

        Returns:
            TextClassifier: The saved model.
        """
        joblib.dump(self, model_path)
        logger.info(f"Model saved to {model_path}")

        return self

    def load_model(self, model_path: str, tokenizer_path: str = None):
        """
        Load the model.

        Args:
            model_path (str): Path to load the model.
            tokenizer_path (str, optional): Path to load the tokenizer. Defaults to None.

        Returns:
            TextClassifier: The loaded model.
        """
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)

