import numpy as np
import pandas as pd
import os 
import sys
import joblib
from prediction.config import config
from prediction.processing.datahandling import load_data, separate_data, split_data,save_pipeline
import prediction.processing.preprocessing as pp
import prediction.pipeline as pipe

def performing_classification():
    dataset=load_data(config.FILE_NAME)
    X,y=separate_data(dataset)
    y = y.apply(lambda x: 1 if x.strip() == "Approved" else 0)
    X_train,X_test,y_train,y_test=split_data(X,y,test_size=0.2)
    test_data=X_test.copy()
    test_data[config.TARGET]=y_test
    test_data.to_csv(os.path.join(config.DATAPATH, config.TEST_FILE), index=False)
    pipe.classification_pipeline.fit(X_train,y_train)
    save_pipeline(pipe.classification_pipeline)


if __name__=='__main__':
    performing_classification()