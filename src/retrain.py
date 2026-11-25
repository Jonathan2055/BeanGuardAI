"""Retraining: fine-tune the SAVED model on the original training images + all uploaded images.

Steps
  1. collect data    (data/train + every uploaded image from the database)
  2. load the saved model (bean_model.keras): this is the pre-trained starting point
  3. fine-tune it with a small learning rate
  4. compare old and new model on the SAME test set and keep the better one
"""
import json
import os
from datetime import datetime, timedelta

from tensorflow import keras

from src import config, database
from src.model import compile_model, evaluate_model, export_inference_model, unfreeze_top_layers
from src.preprocessing import list_images, make_dataset

STATUS_PATH = config.MODELS_DIR / "retrain_status.json"
METRICS_PATH = config.MODELS_DIR / "metrics.json"
EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))


# ---- retraining status (kept in a file, so every API container can read it) ----
def read_status():
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as f:
            return json.load(f)
    return {"state": "idle", "message": "No retraining yet", "updated_at": None}


def write_status(state, message):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w") as f:
        json.dump({"state": state, "message": message,
                   "updated_at": datetime.now().isoformat(timespec="seconds")}, f)


def is_running():
    status = read_status()
    if status["state"] != "running":
        return False
    # a "running" status older than 2 hours means the server stopped during training: ignore it
    return datetime.now() - datetime.fromisoformat(status["updated_at"]) < timedelta(hours=2)


def read_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# ---- the retraining pipeline ----
def run_retraining(epochs=EPOCHS):
    started_at = database.now()
    uploads = database.get_all_uploads()
    pending = database.get_pending_uploads()

    if not pending:
        write_status("idle", "No new images to train on. Upload images first.")
        return

    write_status("running", f"Retraining with {len(pending)} new images ...")
    try:
        class_names = config.load_class_names()

        # 1. data: original training images + every uploaded image (the test set is never used for training)
        train_paths, train_labels = list_images(config.DATA_DIR / "train", class_names)
        val_paths, val_labels = list_images(config.DATA_DIR / "val", class_names)
        test_paths, test_labels = list_images(config.DATA_DIR / "test", class_names)
        train_paths += [u["path"] for u in uploads]
        train_labels += [class_names.index(u["label"]) for u in uploads]

        train_ds = make_dataset(train_paths, train_labels, shuffle=True)
        val_ds = make_dataset(val_paths, val_labels)
        test_ds = make_dataset(test_paths, test_labels)

        # 2. load the saved model and measure it BEFORE retraining
        model = keras.models.load_model(config.MODEL_PATH, compile=False)
        compile_model(model, 1e-5)
        old = evaluate_model(model, test_ds, class_names)

        # 3. fine-tune: small learning rate, stop early when the validation loss stops improving
        unfreeze_top_layers(model)
        compile_model(model, 1e-5)
        stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[stop_early], verbose=2)

        # 4. evaluate again on the same test set; keep the new model only if it is not worse
        new = evaluate_model(model, test_ds, class_names)
        promoted = new["accuracy"] >= old["accuracy"]

        if promoted:
            candidate = config.MODELS_DIR / "candidate.keras"
            model.save(candidate)
            os.replace(candidate, config.MODEL_PATH)                     # swap the file in one step
            export_inference_model(model).save(config.MODEL_H5_PATH)     # refresh the .h5 copy too

            metrics = dict(new)
            metrics["baseline_accuracy"] = read_metrics().get("baseline_accuracy")
            metrics["version"] = read_metrics().get("version", 1) + 1
            metrics["note"] = f"Retrained with {len(uploads)} uploaded images"
            metrics["trained_at"] = database.now()
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)
            message = f"New model saved: accuracy {old['accuracy']} -> {new['accuracy']}"
        else:
            message = (f"The new model was worse ({new['accuracy']} vs {old['accuracy']}), "
                       "so the previous model was kept")

        database.mark_used([u["id"] for u in pending])
        database.add_run(started_at, len(pending), old["accuracy"], new["accuracy"], promoted, message)
        write_status("done", message)

    except Exception as error:
        write_status("failed", f"Retraining failed: {error}")
        database.add_run(started_at, len(pending), None, None, False, f"failed: {error}")
