import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from feature_engine import missing_data_imputers as mdi
from catboost import CatBoostClassifier


def add_time_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    
    df['sine_hr'] = np.sin(df.index.hour/6)
    df['cos_hr'] = np.cos(df.index.hour/6)
    return df

class CategoricalImputer(TransformerMixin, BaseEstimator):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        return self

    def transform(self,X):
        imputer = mdi.CategoricalVariableImputer(variables=self.variables)
        imputer.fit(X)
        return imputer.transform(X)

    
class temporal_featurizer(TransformerMixin, BaseEstimator):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables  = variables
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = add_time_features(X)
        return X

class CustomCatBoostClassifier(CatBoostClassifier):

    def fit(self, X, y=None, **fit_params):
        return super().fit(
            X,
            y=y,
            **fit_params
        )