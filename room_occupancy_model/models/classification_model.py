import joblib
import preprocessors as pp
from add_time_features import add_time_features 

class prediction_model():
    def __init__(self):
        self._model = joblib.load('prediction_pipeline.sav')

    def predict(self, X):
        return self._model.predict(X)

if __name__ == '__main__':
    model = prediction_model()

