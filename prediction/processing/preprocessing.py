from sklearn.base import BaseEstimator, TransformerMixin
from prediction.config import config
from pathlib import Path
import numpy as np

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop=columns_to_drop
    
    def fit(self, X,y=None):
        return self
    
    def transform(self, X):
        X=X.drop(self.columns_to_drop, axis=1)
        return X


class AddNewColumn(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_add=None):
        self.columns_to_add=columns_to_add
        self.new_column= config.NEW_FEATURE_ADD
    
    def fit(self, X,y=None):
        return self
    
    def transform(self, X):
        X[self.new_column]=X[self.columns_to_add].sum(axis=1)
        return X

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variable=None):
        self.variable = variable  # dict: {column_name: [positive_values]}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column_name, positive_values in self.variable.items():
            X[column_name] = X[column_name].apply(
                lambda x: 1 if str(x).strip() in positive_values else 0
            )
        return X

class LogTranssformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_variable):
        self.log_variable=log_variable
    def fit(self,X,y=None):
        return self
    def transform(self, X):
        X=X.copy()
        for col in self.log_variable:
            X[col]=np.log(X[col])
        return X
