from room_occupancy_model.preprocessors.classifier_pipeline import preprocess_pipe
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loguru import logger
import numpy as np
import pandas as pd
import joblib
import os
import dotenv

dotenv.load_dotenv()

pkg_root_dir = os.environ.get('room_occupancy_root_dir')

data_file_path = os.path.join(pkg_root_dir, 'data/external/datatest.txt')
logger.info('Reading dataset...')
try:
    data_raw = pd.read_csv(data_file_path, index_col=1, parse_dates=True)
except:
    logger.info('Exception data could not be read')
else:
    logger.info('Data read succesfully in memory...')



logger.info('Splitting the data into traing and test set...')
X_train, X_test, y_train, y_test = train_test_split(data_raw.drop(columns=['Occupancy'], axis=1), data_raw['Occupancy'], test_size=0.2, stratify=data_raw['Occupancy'])
logger.info('Data split into training and test set, test size is 20% of training data')
logger.info('Wrting testing data into data folder...')
X_test.to_pickle(os.path.join(pkg_root_dir,'tests/X_test.pkl'))
y_test.to_pickle(os.path.join(pkg_root_dir,'tests/y_test.pkl'))

logger.info('Training the pipeline with training data')
preprocess_pipe.fit(X_train, y_train)
logger.info('Training finished')

pickle.dump(preprocess_pipe, open(os.path.join(pkg_root_dir,'room_occupancy_model/models/prediction_pipeline.sav'), 'wb'))

predicted_values = preprocess_pipe.predict(X_test)
logger.info('Wrting predicted values for test data from trained model')
pickle.dump(predicted_values, 
    open(os.path.join(pkg_root_dir, 'tests/model_predictions.pkl'), 'wb'))


training_report = classification_report(y_test, predicted_values)
logger.info('Classification report for the trained model...\n')
logger.info('\n' + training_report)


