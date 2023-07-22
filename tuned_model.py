import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from config import MODEL_PATH


class TunedModel:
    def __init__(self):
        self._model: Model = load_model(str(MODEL_PATH))

    def predict_single(self, X: np.ndarray, *, threshold: float = 0.995) -> tuple[bool, float]:
        y_pred_logit = self._model.predict([X])
        y_pred_proba = tf.sigmoid(y_pred_logit).numpy().flatten()
        y_pred = y_pred_proba >= threshold
        return y_pred[0], y_pred_proba[0]
