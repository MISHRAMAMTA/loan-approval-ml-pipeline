import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os
import sys

from prediction.config import config  
from prediction.processing.datahandling import load_data, separate_data, split_data,load_pipeline

classification_pipeline = load_pipeline(config.MODEL_NAME)



def generate_predictions():
    test_data = load_data(config.TEST_FILE)
    X,y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    output = np.where(pred==1,'Approved','Not Approved')
    print(output)
    return output


if __name__=='__main__':
    generate_predictions()