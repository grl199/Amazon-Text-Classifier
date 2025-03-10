import gzip
import json
import re
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import classification_report,accuracy_score
import logging


def read_config(file):
    '''
    Read the config file
    '''
    with open(file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_logger(config = None):
    '''
    Set the logger
    '''
    
    logging.basicConfig(
        level = config.get('log_level', 'INFO'),
        format= config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    logger = logging.getLogger('main')

    return logger


def load_data(path, step=1000):
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for i, l in enumerate(g):
            if i%step ==0:
                yield json.loads(l)


def preprocess_image_urls(X,col_name = 'image'):
    """Extrae palabras clave de URLs en una columna 'image'"""
    if col_name in X.columns:
        X = X.copy()
        # Asegurarse de que el campo col_name no sea nulo y sea de tipo string
        X[col_name] = X[col_name].fillna('').astype(str)
        # Aplicar la extracción de palabras clave
        X[col_name] = X[col_name].apply(lambda x: extract_url_keywords(x))
        # Si después de procesar queda vacío, agregar un valor por defecto
        X[col_name] = X[col_name].apply(lambda x: 'no_image_url' if x.strip() == '' else x)

    return X

def preprocess_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_array_of_texts(x, stop_words):
    return [preprocess_text(text,stop_words) for text in x]


def extract_url_keywords(url):
    """
    Extrae palabras clave significativas de una URL.
    
    Args:
        url (str): URL a procesar
        
    Returns:
        str: Palabras clave extraídas de la URL
    """
    if not isinstance(url, str) or not url:
        return "no_url"
    
    # Eliminar protocolo (http://, https://)
    url = re.sub(r'https?://', '', url)
    
    # Eliminar www. si existe
    url = re.sub(r'www\.', '', url)
    
    # Eliminar extensiones comunes y parámetros
    url = re.sub(r'\.(com|org|net|edu|gov|co|io|html?|php|aspx).*$', '', url)
    
    # Separar por caracteres no alfanuméricos
    keywords = re.split(r'[^a-zA-Z0-9]', url)
    
    # Filtrar palabras vacías y convertir a minúsculas
    keywords = [word.lower() for word in keywords if word and len(word) > 1]
    
    # Si no se encontraron palabras clave, devolver un valor predeterminado
    if not keywords:
        return "unknown_url"
    
    return " ".join(keywords)

def create_combined_text(df, columns, name_of_output_columns='text'):
    """
    Combina el texto de múltiples columnas en una sola, procesando URLs si están presentes.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar
        columns (list): Lista de columnas a combinar
        name_of_output_columns (str): Nombre de la columna de salida
        
    Returns:
        pd.DataFrame: DataFrame con la nueva columna de texto combinado
    """
    df[name_of_output_columns] = df.apply(lambda row: " ".join([
        extract_url_keywords(row.get(col, "")) if col == 'image' else
        " ".join(row.get(col, [])) if isinstance(row.get(col, []), list) else str(row.get(col, ""))
        for col in columns
    ]), axis=1)
    return df



def extract_price(X):
    def convert_price(price):
        # Remove any non-numeric characters except for the decimal point and dash
        if not isinstance(price, str):
            return np.nan
        price = re.sub(r'[^\d.-]', '', price)
        # Split the price range if it exists
        try:
            if '-' in price:
                low, high = map(float, price.split('-'))
                return (low + high) / 2
            else:
                return float(price)
        except:
            return np.nan
    
    if isinstance(X, pd.DataFrame):
        return X.applymap(convert_price)
    else:
        return X.apply(convert_price).to_frame()
    


