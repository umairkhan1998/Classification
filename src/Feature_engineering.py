import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn



# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('Feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'Feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def apply_tfidf(df, content_column):
    try:
        logger.debug('Applying TF-IDF...')
        
        df[content_column] = df[content_column].fillna("")

        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(df[content_column])
        
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        tfidf_df["annotation"] = df["annotation"].values  # Ensure it's copied correctly
        
        logger.debug("TF-IDF transformation completed successfully.")
        return tfidf_df
    except ValueError as e:
        logger.error('Failed to perform TFIDF: %s', e)
        return None
def imbalance_data(vectorized_data):
    try:
        # Separate features and target
        X = vectorized_data.drop(columns=["annotation"])  # Features
        y = vectorized_data["annotation"]  # Target
        
        # Ensure target is integer
        y = y.astype(int)
        # Apply SMOTE
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name="annotation")
        X_resampled["annotation"] = y_resampled
        return X_resampled
    
    except ValueError as e:
        logger.error("Data balanceing isn't performed.")
  


mlflow.autolog()
mlflow.set_experiment("Feature_engineering")

def main():
    try:
      with mlflow.start_run():
        # Load data
        data_path = 'data/interim/train_processed.csv'
        Load_data = pd.read_csv(data_path)
        logger.debug("Data loaded successfully.")

        # Apply TF-IDF
        vectorized_data = apply_tfidf(Load_data, "content")
        if vectorized_data is None:
            logger.error("TF-IDF transformation failed. Exiting.")
            return
        #Applying Imbalance function on data
        vectorized_data =  imbalance_data(vectorized_data)
        logger.debug("Handling data completed successfully.")

        # Save processed data
        output_path = "./data/Vectorized"
        os.makedirs(output_path, exist_ok=True)
        vectorized_data.to_csv(os.path.join(output_path, "Vectorize_data.csv"), index=False)
        logger.debug('Vectorized data saved to %s', output_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
