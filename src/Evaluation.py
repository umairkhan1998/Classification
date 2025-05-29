import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import logging
import pickle
import json
import mlflow
import mlflow.sklearn



# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'Evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    try:
        logger.debug("Evaluating the model...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # For ROC curve
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)  # Return as dictionary
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        logger.info(f"Model Accuracy: {accuracy:.2f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{cr}")
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save ROC curve plot
        roc_plot_path = os.path.join('Results', 'roc_curve.png')
        os.makedirs(os.path.dirname(roc_plot_path), exist_ok=True)
        plt.savefig(roc_plot_path)
        logger.debug(f"ROC curve saved to {roc_plot_path}")
        plt.show()
        
        # Return metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),  # Convert numpy array to list for JSON serialization
            'classification_report': cr,
            'roc_auc': roc_auc
        }
        return metrics
        
    except Exception as e:
        logger.error(f"Error while evaluating the model: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

mlflow.autolog()
mlflow.set_experiment("Evaluation")


def main():
    try:
      with mlflow.start_run():   
        # Load test data
        x_test = pd.read_csv(r'C:\Users\umair\OneDrive\Desktop\cyber_bulling\data\test_data\X_test.csv')
        y_test = pd.read_csv(r'C:\Users\umair\OneDrive\Desktop\cyber_bulling\data\test_data\y_test.csv')  # Ensure y_test is a Series
        
        # Load model
        model_path = 'models\model.pkl'
        model = load_model(model_path)
        
        # Evaluate model
        metrics = evaluate_model(model, x_test, y_test)
        
        # Save metrics
        results_dir = 'Results'
        metrics_file_path = os.path.join(results_dir, 'metrics.json')
        save_metrics(metrics, metrics_file_path)
        
        
        # Log metrics
        mlflow.log_metric('accuracy', metrics['accuracy'])
        mlflow.log_metric('roc_auc', metrics['roc_auc'])

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")

if __name__ == '__main__':
    main()