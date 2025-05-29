import nltk
import re, string
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
import logging
import tensorflow as tf
import pandas as pd
from nltk.stem.porter import PorterStemmer
import mlflow
import mlflow.sklearn



# Download required NLTK resources
nltk.download('stopwords')
stop = stopwords.words('english')
regex = re.compile('[%s]' % re.escape(string.punctuation))

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def custom_standardization(input_data):
    """
    Custom text preprocessing pipeline using TensorFlow.
    """
    # Convert input to TensorFlow string tensor
    input_data = tf.convert_to_tensor(input_data, dtype=tf.string)

    # Convert to lowercase
    lowercase = tf.strings.lower(input_data)

    # Remove URLs
    stripped_urls = tf.strings.regex_replace(lowercase, r"https?://\S+|www\.\S+", "")

    # Remove email addresses
    stripped_emails = tf.strings.regex_replace(stripped_urls, r"\S*@\S*\s?", "")

    # Remove text in angular brackets (usually HTML tags)
    stripped_brackets = tf.strings.regex_replace(stripped_emails, r"<.*?>+", "")

    # Remove any square brackets while leaving text inside
    stripped_brackets = tf.strings.regex_replace(stripped_brackets, r"\[|\]", "")

    # Remove numbers and words containing digits
    stripped_digits = tf.strings.regex_replace(stripped_brackets, r"\w*\d\w*", "")

    # Replace multiple whitespaces with a single space
    stripped_whitespace = tf.strings.regex_replace(stripped_digits, r"\s+", " ")

    # Remove non-alphabet characters
    cleaned_text = tf.strings.regex_replace(stripped_whitespace, r"[^a-zA-Z\s]+", "")

    return cleaned_text.numpy().decode('utf-8')  # Convert TensorFlow tensor back to string

def preprocess_text_column(df, column_name):
    """
    Applies text preprocessing to a Pandas DataFrame column using TensorFlow operations.
    """
    logger.debug(f"Processing column: {column_name}")

    # Apply custom_standardization to each row in the column
    df[column_name] = df[column_name].apply(custom_standardization)
    return df

mlflow.autolog()
mlflow.set_experiment("data_preprocessing")

def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
      with mlflow.start_run():
        # Load the data
        prepro_data = pd.read_csv('data/raw/Ready_data.csv')
        logger.debug('Data loaded successfully')

        # Ensure 'text' column is of type string
        prepro_data["content"] = prepro_data["content"].astype(str)
       

        # Apply text preprocessing
        prepro_data = preprocess_text_column(prepro_data, "content")
        

        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        prepro_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
       

        logger.debug('Processed data saved to %s', data_path)
    
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
