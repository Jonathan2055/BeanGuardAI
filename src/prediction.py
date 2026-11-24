"""Prediction: load the saved model and classify one image."""
import os
import threading

import numpy as np
from tensorflow import keras

from src.config import MODEL_PATH, load_class_names
from src.preprocessing import image_to_array, validate_image

_model = None
_model_time = None
_lock = threading.Lock()


def get_model():
    """Load the model once. If the file changes (after retraining), load the new one."""
    global _model, _model_time
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Train and save it first (run the notebook).")
    modified = os.path.getmtime(MODEL_PATH)
    if _model is None or modified != _model_time:
        with _lock:
            if _model is None or modified != _model_time:
                _model = keras.models.load_model(MODEL_PATH, compile=False)
                _model_time = modified
    return _model


def predict_image(file_bytes):
    """Takes the bytes of an image file. Returns the predicted class and all probabilities."""
    class_names = load_class_names()
    image = validate_image(file_bytes)
    probabilities = get_model()(image_to_array(image), training=False).numpy()[0]
    best = int(np.argmax(probabilities))
    return {
        "class": class_names[best],
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probabilities)},
    }
