from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys
from prediction.config import config
import prediction.processing.preprocessing as pp
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline=Pipeline(
    [
        ('AddNewColumn',pp.AddNewColumn(columns_to_add = config.FEATURE_TO_ADD)),
        ('DropColumn', pp.DropColumn(columns_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder',pp.CustomLabelEncoder(variable=config.FEATURES_TO_ENCODE)),
        ('LogTransform',pp.LogTranssformer(log_variable=config.LOG_FEATURES)),
        ('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)