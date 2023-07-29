import numpy as np
import tensorflow as tf

from config import CONFIDENCE, MODEL_PATH


class TunedModel:
    def __init__(self):
        from model import get_model
        self._model = get_model(imagenet_weights=False)
        self._model.load_weights(str(MODEL_PATH))

    def predict_single(self, X: np.ndarray, *, threshold: float = CONFIDENCE) -> tuple[bool, float]:
        with tf.device('/CPU:0'):  # force CPU to better understand real performance
            y_pred_proba = self._model.predict(X[np.newaxis, ...]).flatten()
            y_pred = y_pred_proba > threshold
            return y_pred[0], float(y_pred_proba[0])
