import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from prediction.config import config

def load_data(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    dataset = pd.read_csv(filepath)
    dataset.columns = [c.strip() for c in dataset.columns]
    return dataset

def separate_data(data):
    X = data.drop(config.TARGET, axis=1)
    y = data[config.TARGET]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model saved successfully at {save_path}")

def load_pipeline(pipeline_to_load):
    save_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_to_load)
    model_loaded = joblib.load(save_path)
    print(f"Model '{pipeline_to_load}' loaded successfully.")
    return model_loaded