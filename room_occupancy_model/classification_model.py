import joblib
import os
import dotenv
import sys
from room_occupancy_model.preprocessors import classifier_pipeline
from room_occupancy_model.preprocessors import pipe_preprocessors as pp

dotenv.load_dotenv()


class prediction_model():
    def __init__(self):
        home_dir = os.environ.get("pkg_root_dir")
        model_path = os.path.join(home_dir, 'room_occupancy_model/models/prediction_pipeline.sav')
        self._model = joblib.load(model_path)


    def predict(self, X):
        return self._model.predict(X)



