import numpy as np

def add_time_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    
    df['sine_hr'] = np.sin(df.index.hour/6)
    df['cos_hr'] = np.cos(df.index.hour/6)
    return df