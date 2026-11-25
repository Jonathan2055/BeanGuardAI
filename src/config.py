"""Settings shared by the notebook, the retraining code, the API and the UI."""
import json
from pathlib import Path

# Project root = the folder that contains src/
# BASE_DIR is detected from this file, so it works on Kaggle and your laptop.
# Make sure PROJECT_ROOT points to your local BeanGuardAI folder when using a laptop.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"  # saved model and evaluation metrics
MODEL_PATH = MODELS_DIR / "bean_model.keras"
MODEL_H5_PATH = MODELS_DIR / "bean_model.h5"

IMG_SIZE = (224, 224)     # every image is resized to 224 x 224
BATCH_SIZE = 32           # the model looks at 32 images at a time


def load_class_names():
    """The class order saved in data/class_names.json by the notebook."""
    with open(DATA_DIR / "class_names.json") as f:
        return json.load(f)
