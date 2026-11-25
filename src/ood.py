"""Input check: is the uploaded photo a bean leaf at all?

The model only knows 3 classes, so for a photo of a cat it would still answer "healthy" or "rust".
To catch this we compare the photo with the bean leaves we trained on:

1. For every training image we take the model's internal feature vector (1280 numbers, the layer
   just before the final classification layer).
2. We average these vectors per class (a "class centre").
3. For a new photo we measure how similar its feature vector is to the closest class centre
   (cosine similarity). Unrelated images can still resemble bean-leaf features.
4. The limit (threshold) is calibrated from validation leaves to target about 99% acceptance.

Commands (run from the project folder, after training):
    python -m src.ood build              create models/ood_stats.json from the saved model
    python -m src.ood check <folder>     test a folder of NON-bean photos and the bean test images
"""

import json
import os
import sys
import threading
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src import config, database
from src.prediction import get_model
from src.preprocessing import image_to_array, list_images, make_dataset, validate_image

STATS_PATH = config.MODELS_DIR / "ood_stats.json"
PERCENTILE = float(os.getenv("OOD_PERCENTILE", "1"))  # % of real validation leaves that may be rejected
CONFIDENCE_LIMIT = 0.6

_lock = threading.Lock()
_feature_model, _feature_model_for = None, None
_stats, _stats_time = None, None


def make_feature_model(model):
    """A copy of the model that returns the 1280 internal features AND the 3 probabilities."""
    pooling = next(layer for layer in model.layers if isinstance(layer, layers.GlobalAveragePooling2D))
    return keras.Model(model.input, [pooling.output, model.output])


def get_feature_model(model):
    global _feature_model, _feature_model_for
    with _lock:
        if _feature_model_for is not model:
            _feature_model, _feature_model_for = make_feature_model(model), model
        return _feature_model


def _prepare(features, mean):
    """Centre the features (subtract the average training vector) and scale them to length 1."""
    centred = features - mean
    return centred / (np.linalg.norm(centred, axis=-1, keepdims=True) + 1e-9)


def _embed(feature_model, paths):
    dataset = make_dataset(paths, [0] * len(paths))
    return feature_model.predict(dataset, verbose=0)[0]


def build_stats(
    model,
    class_names,
    percentile=PERCENTILE,
    output_path=STATS_PATH,
    owner_id=None,
):
    """Compute the class centres and the threshold, and save them in models/ood_stats.json."""
    feature_model = make_feature_model(model)

    train_paths, train_labels = list_images(config.DATA_DIR / "train", class_names)
    if owner_id is not None:
        database.init_db()
        for upload in database.get_all_uploads(owner_id):
            if os.path.exists(upload["path"]):
                train_paths.append(upload["path"])
                train_labels.append(class_names.index(upload["label"]))
    val_paths, _ = list_images(config.DATA_DIR / "val", class_names)

    train_features = _embed(feature_model, train_paths)
    val_features = _embed(feature_model, val_paths)

    mean = train_features.mean(axis=0)
    train_ready = _prepare(train_features, mean)
    labels = np.array(train_labels)
    centres = np.stack([train_ready[labels == k].mean(axis=0) for k in range(len(class_names))])
    centres = centres / np.linalg.norm(centres, axis=1, keepdims=True)

    val_similarity = (_prepare(val_features, mean) @ centres.T).max(axis=1)
    stats = {
        "threshold": round(float(np.percentile(val_similarity, percentile)), 4),
        "percentile": percentile,
        "val_min": round(float(val_similarity.min()), 4),
        "val_median": round(float(np.median(val_similarity)), 4),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "mean": mean.round(5).tolist(),
        "centres": centres.round(5).tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary, "w") as f:
        json.dump(stats, f)
    os.replace(temporary, output_path)
    return stats


def load_stats():
    """Read the saved statistics (and re-read them if the file changed after retraining)."""
    global _stats, _stats_time
    if not STATS_PATH.exists():
        return None
    modified = os.path.getmtime(STATS_PATH)
    if _stats is None or modified != _stats_time:
        with open(STATS_PATH) as f:
            data = json.load(f)
        data["mean"], data["centres"] = np.array(data["mean"]), np.array(data["centres"])
        _stats, _stats_time = data, modified
    return _stats


def analyse(file_bytes):
    """Check that the photo looks like a bean leaf, then classify it."""
    class_names = config.load_class_names()
    model = get_model()
    image = validate_image(file_bytes)
    features, probabilities = get_feature_model(model)(image_to_array(image), training=False)
    probabilities = probabilities.numpy()[0]
    best = int(np.argmax(probabilities))

    result = {
        "class": class_names[best],
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(class_names, probabilities)},
        "uncertain": bool(probabilities[best] < CONFIDENCE_LIMIT),
        "is_bean_leaf": None,
        "similarity": None,
        "threshold": None,
    }

    stats = load_stats()
    if stats is None:
        raise FileNotFoundError("Input validation statistics are missing. Run the notebook first.")

    ready = _prepare(features.numpy()[0], stats["mean"])
    similarity = float((stats["centres"] @ ready).max())
    threshold = float(os.getenv("OOD_THRESHOLD", stats["threshold"]))
    result.update(
        similarity=round(similarity, 4),
        threshold=round(threshold, 4),
        is_bean_leaf=bool(similarity >= threshold),
    )

    if not result["is_bean_leaf"]:
        return {
            "rejected": True,
            "is_bean_leaf": False,
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "message": "This does not look like a bean leaf. Take a close, well-lit photo of a single leaf.",
        }
    return result


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    names = config.load_class_names()

    if command == "build":
        info = build_stats(keras.models.load_model(config.MODEL_PATH, compile=False), names)
        print(f"Saved {STATS_PATH}")
        print(
            f"threshold {info['threshold']}  "
            f"(validation leaves: lowest {info['val_min']}, median {info['val_median']})"
        )

    elif command == "check" and len(sys.argv) > 2:
        def scan(paths):
            sims = []
            for p in paths:
                with open(p, "rb") as f:
                    result = analyse(f.read())
                    sims.append(result)
            return sims

        folder = sys.argv[2]
        others = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        results = scan(others)
        print("NON-bean photos:")
        for p, r in zip(others, results):
            print(
                f"  {os.path.basename(p):30s} similarity {r['similarity']}  ->",
                "accepted (wrong!)" if r["is_bean_leaf"] else "rejected",
            )
        rejected = sum(not r["is_bean_leaf"] for r in results)
        print(f"  rejected {rejected} of {len(results)}")
        test_paths, _ = list_images(config.DATA_DIR / "test", names)
        test_results = scan(test_paths)
        accepted = sum(r["is_bean_leaf"] for r in test_results)
        print(f"Bean test images accepted: {accepted} of {len(test_results)}")
    else:
        print(__doc__)
