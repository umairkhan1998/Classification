import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
import os
import logging
import mlflow
import mlflow.sklearn



# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model_Building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'Model_Building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_data(data_path):
    try:
        logger.debug("Data loading...")
        data = pd.read_csv(data_path)
        logger.debug("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error while loading data: {e}")
        raise
    return data

def train_test(Data, target):
    try:
        logger.debug("Splitting data into train and test sets...")
        X = Data.drop(columns=[target])
        y = Data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        logger.debug("Data split successfully")
    except Exception as e:
        logger.error(f"Error while splitting data: {e}")
        raise
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    try:
        logger.debug("Training the model...")
        model = RandomForestClassifier(verbose=True)
        model.fit(X_train, y_train)
        logger.debug("Model trained successfully")
    except Exception as e:
        logger.error(f"Error while training the model: {e}")
        raise
    return model

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise
def save_test_data(X_test, y_test, test_data_save_dir):
    """
    Save the test data (X_test and y_test) to CSV files.
    """
    try:
        logger.debug("Saving test data...")
        os.makedirs(test_data_save_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        X_test_path = os.path.join(test_data_save_dir, 'X_test.csv')
        y_test_path = os.path.join(test_data_save_dir, 'y_test.csv')
        
        X_test.to_csv(X_test_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        
        logger.info(f"X_test saved to {X_test_path}")
        logger.info(f"y_test saved to {y_test_path}")
    except Exception as e:
        logger.error(f"Error while saving test data: {e}")
        raise



 
mlflow.autolog()
mlflow.set_experiment("model_building")


def main():
    try:
      with mlflow.start_run():  
        data_path = 'data/Vectorized/Vectorize_data.csv'
        Data = get_data(data_path)
        
        X_train, X_test, y_train, y_test = train_test(Data, 'annotation')
        
        clf = train_model(X_train, y_train)
        
        model_save_path = 'models/model.pkl'

        save_model(clf, model_save_path)

         
        # Save the test data
        test_data_save_dir = 'data/test_data'
        save_test_data(X_test, y_test, test_data_save_dir)
        
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

if __name__ == '__main__':
    main()