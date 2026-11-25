"""Sanity check: does the saved model reproduce the test score recorded in models/metrics.json?

Run from the project folder:   python -m src.check_model
If the two accuracies match, the model file, the data/test folder and the class order are all correct.
"""
import json

from tensorflow import keras

from src import config
from src.model import compile_model, evaluate_model
from src.preprocessing import list_images, make_dataset

class_names = config.load_class_names()
paths, labels = list_images(config.DATA_DIR / "test", class_names)
print(f"Test images found: {len(paths)}")

model = keras.models.load_model(config.MODEL_PATH, compile=False)
compile_model(model, 1e-5)
result = evaluate_model(model, make_dataset(paths, labels), class_names)

with open(config.MODELS_DIR / "metrics.json") as f:
    saved = json.load(f)

print(f"Accuracy now          : {result['accuracy']}")
print(f"Accuracy in metrics   : {saved['accuracy']}")
print("MATCH" if abs(result["accuracy"] - saved["accuracy"]) < 0.005 else "DIFFERENT: check data/test and the model file")
