"""Prediction: load the saved model and classify one image."""
import os
import threading

import numpy as np
from tensorflow import keras

from src.config import load_class_names
from src.model_versions import INITIAL_VERSION, model_path
from src.preprocessing import image_to_array, validate_image

_models = {}
_lock = threading.Lock()


def get_model(version=INITIAL_VERSION):
    """Load an account-selected immutable model once and reuse it for predictions."""
    path = model_path(version)
    if not path.exists():
        raise FileNotFoundError("Model not found. Train and save it first (run the notebook).")
    modified = os.stat(path).st_mtime_ns
    cached = _models.get(version)
    if cached is None or cached[0] != modified:
        with _lock:
            cached = _models.get(version)
            if cached is None or cached[0] != modified:
                cached = (modified, keras.models.load_model(path, compile=False))
                _models[version] = cached
    return cached[1]


def predict_image(file_bytes, version=INITIAL_VERSION):
    """Takes the bytes of an image file. Returns the predicted class and all probabilities."""
    class_names = load_class_names()
    image = validate_image(file_bytes)
    probabilities = get_model(version)(image_to_array(image), training=False).numpy()[0]
    best = int(np.argmax(probabilities))
    return {
        "class": class_names[best],
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probabilities)},
        "model_version": version,
    }
