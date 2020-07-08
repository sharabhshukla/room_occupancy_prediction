import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from room_occupancy_model.preprocessors import pipe_preprocessors as pp

preprocess_pipe = Pipeline(
    [
        ('temporal_feature_add', pp.temporal_featurizer(None)),

        ("estimator", pp.CustomCatBoostClassifier(early_stopping_rounds=5, silent=True))
    ]
)


