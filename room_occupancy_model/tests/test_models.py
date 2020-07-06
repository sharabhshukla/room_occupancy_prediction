import pytest
import pickle
import sys
import os
from sklearn.metrics import accuracy_score
from loguru import logger

sys.path.append(os.environ.get("room_occupancy_root_dir"))
sys.path.append(os.path.join(os.environ.get("room_occupancy_root_dir"), 'room_occupancy_model', 'models'))
home_dir = os.environ.get("room_occupancy_root_dir")
from room_occupancy_model.models.classification_model import prediction_model

model = prediction_model()

with open('test_file.txt', 'w') as f:
    f.write("Some shit")

test_data = pickle.load(open(os.path.join(home_dir, 'room_occupancy_model/tests/X_test.pkl'), 'rb'))
model_predicted_data = pickle.load(open(os.path.join(home_dir, 'room_occupancy_model/tests/model_predictions.pkl'), 'rb'))
true_predictions = pickle.load(open(os.path.join(home_dir, 'room_occupancy_model/tests/y_test.pkl'), 'rb'))

def test_model():
    y_pred = model.predict(test_data)
    assert list(model_predicted_data) == list(y_pred)

def test_model_accuracy():
    y_pred = model.predict(test_data)
    model_accuracy = accuracy_score(true_predictions,y_pred) 
    logger.info('Model accuracy is around {}'.format(model_accuracy))
    assert model_accuracy == pytest.approx(0.99, 0.01)