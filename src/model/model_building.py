import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import xgboost as xgb
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Preprocessing function — HARUS SAMA dengan Flask
def preprocessing_comment(comment):
    try:
        comment = comment.lower().strip()

        comment = re.sub(r'\n', ' ', comment)
        
        comment = re.sub('[^A-Za-z0-9\s!?.,]', '', comment)
        
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        
        words = [word for word in comment.split() if word not in stop_words]
        
        lemmatizer = WordNetLemmatizer()
        
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        
        return comment


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            
            params = yaml.safe_load(file)
        
        logger.debug('Parameters retrieved from %s', params_path)
        
        return params
    except Exception as e:
        logger.error('Failed to load parameters: %s', e)
        
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        
        df.fillna('', inplace=True)
        
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        
        return df
    
    except Exception as e:
        logger.error('Error loading data: %s', e)
        
        raise


def apply_bow(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    try:
        # Preprocess comments here to ensure consistency with inference
        train_data['Comment'] = train_data['Comment'].apply(preprocessing_comment)

        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train = train_data['Comment'].values
        y_train = train_data['Sentiment'].values

        # Bag of Words transformation
        X_train_bow = vectorizer.fit_transform(X_train)

        logger.debug(f"BoW transformation complete. Shape: {X_train_bow.shape}")

        # Save vectorizer
        with open(os.path.join(get_root_directory(), 'bow_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        return X_train_bow, y_train
    except Exception as e:
        logger.error('Error in apply_bow: %s', e)
        raise


def train_xgb(X_train, y_train, num_class, learning_rate, max_depth, n_estimators, colsample_bylevel, colsample_bytree, gamma, reg_alpha, reg_lambda, subsample):
    try:
        best_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_class,
            colsample_bylevel= colsample_bylevel,
            colsample_bytree= colsample_bytree,
            gamma= gamma,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=2,
            n_estimators=n_estimators,
            reg_alpha= reg_alpha,
            reg_lambda= reg_lambda,
            subsample=subsample,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        best_model.fit(X_train, y_train)
        logger.debug('XGBoost model training completed')
        return best_model
    except Exception as e:
        logger.error('Error in train_xgb: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        root_dir = get_root_directory()

        # Load params
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])
        learning_rate = params['model_building']['learning_rate']
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        num_class = params['model_building']['num_class']
        colsample_bylevel = params['model_building']['colsample_bylevel']
        colsample_bytree = params['model_building']['colsample_bytree']
        gamma = params['model_building']['gamma']
        reg_alpha = params['model_building']['reg_alpha']
        reg_lambda = params['model_building']['reg_lambda']
        subsample = params['model_building']['subsample']

        # Load data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Map labels to numbers
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        train_data['Sentiment'] = train_data['Sentiment'].map(label_mapping)

        # Apply BoW + Preprocessing
        X_train_bow, y_train = apply_bow(train_data, max_features, ngram_range)

        # Train model
        best_model = train_xgb(
            X_train_bow, 
            y_train, 
            learning_rate, 
            max_depth, 
            n_estimators,
            num_class,
            colsample_bylevel,
            colsample_bytree,
            gamma,
            reg_alpha,
            reg_lambda,
            subsample
            )

        # Evaluate model
        y_pred_train = best_model.predict(X_train_bow)
        report = classification_report(y_train, y_pred_train, target_names=['negative', 'neutral', 'positive'])
        print("📊 Classification Report:\n", report)

        # Save model
        save_model(best_model, os.path.join(root_dir, 'xgb_model.pkl'))

    except Exception as e:
        logger.error('Failed to build model: %s', e)
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
