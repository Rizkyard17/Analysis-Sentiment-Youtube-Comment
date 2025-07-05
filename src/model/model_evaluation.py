import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> CountVectorizer:
    """Load the saved Bow vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('Bow vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("file:///D:/PROYEK/MACHINE LEARNING/Ansen_yt/Analysis-Sentiment-Youtube-Comment/mlflow_logs_runs")

    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run() as run:
        try:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters - ensure they're serializable
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
            
            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'xgb_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Mapping label test data: pastikan numerik sama seperti data training
            label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
            test_data['Sentiment'] = test_data['Sentiment'].map(label_mapping)

            # Prepare test data
            X_test_bow = vectorizer.transform(test_data['Comment'].values)
            y_test = test_data['Sentiment'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_bow.toarray()[:5], columns=vectorizer.get_feature_names_out())

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_bow[:5]))

            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "xgb_model",
                signature=signature,
                input_example=input_example
            )

            # Save model info
            model_path = "xgb_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_bow, y_test)

            # Log classification report metrics for the test data with proper error handling
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    safe_label = str(label).replace(" ", "_").replace("/", "_")

                    # Extract metrics with safer handling
                    precision = metrics.get('precision')
                    recall = metrics.get('recall')
                    f1 = metrics.get('f1-score')

                    # Log only if all metrics are valid numbers
                    if precision is not None and isinstance(precision, (int, float)) and not np.isnan(precision):
                        mlflow.log_metric(f"test_{safe_label}_precision", float(precision))
                    
                    if recall is not None and isinstance(recall, (int, float)) and not np.isnan(recall):
                        mlflow.log_metric(f"test_{safe_label}_recall", float(recall))
                    
                    if f1 is not None and isinstance(f1, (int, float)) and not np.isnan(f1):
                        mlflow.log_metric(f"test_{safe_label}_f1_score", float(f1))

            # Log confusion matrix
            log_confusion_matrix(cm, "Test_Data")

            # Add important tags
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()