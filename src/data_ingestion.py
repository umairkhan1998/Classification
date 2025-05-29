import pandas as pd
import os
import logging
import yaml
import mlflow
import mlflow.sklearn


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_json(data_url, lines= True,orient='columns')
        logger.debug('Data loaded from %s', data_url)
        return df
    except ValueError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by converting 'annotation' labels to binary format."""
    try:
        for i in range(len(df)):  # Iterate over DataFrame rows
            if isinstance(df.iloc[i]['annotation'], dict):  # Ensure annotation is a dictionary
                if 'label' in df.iloc[i]['annotation'] and df.iloc[i]['annotation']['label']:
                    df.at[i, 'annotation'] = 1 if df.iloc[i]['annotation']['label'][0] == '1' else 0

        if 'extras' in df.columns:
            df.drop(['extras'], axis=1, inplace=True)
                       
        logger.debug('Data preprocessing completed')
        return df
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise  

def save_data(ingested_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        ingested_data.to_csv(os.path.join(raw_data_path, "Ready_data.csv"), index=False)
        logger.debug('data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

mlflow.autolog()
mlflow.set_experiment("data_ingestion.py")

def main():
    try:
     with mlflow.start_run():
        url = 'https://drive.google.com/uc?export=download&id=12fBlhsa5GIdtme1jT3KlPPIgIdjzqhv1'
        df = load_data(data_url=url)
        final_df = preprocess_data(df)

        
        
        save_data(final_df, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()   