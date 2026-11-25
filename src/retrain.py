"""Fine-tune the saved model using the requesting account's uploaded images."""
import json
import logging
import os
import threading
from datetime import datetime

from tensorflow import keras

from src import config, database
from src.model import compile_model, evaluate_model, export_inference_model, unfreeze_top_layers
from src.preprocessing import list_images, make_dataset

STATUS_PATH = config.MODELS_DIR / "retrain_status.json"
METRICS_PATH = config.MODELS_DIR / "metrics.json"
EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))
HEARTBEAT_INTERVAL = 20
logger = logging.getLogger(__name__)


def _renew_retraining_lease(owner_id, job_id, stop_event):
    while not stop_event.wait(HEARTBEAT_INTERVAL):
        try:
            if not database.renew_retraining(owner_id, job_id):
                logger.error("Retraining lease was lost for job %s.", job_id)
                return
        except Exception:
            logger.exception("Could not renew retraining lease for job %s.", job_id)
            return


def read_status():
    if STATUS_PATH.exists():
        with open(STATUS_PATH) as status_file:
            return json.load(status_file)
    return {"state": "idle", "message": "No retraining yet", "updated_at": None}


def write_status(state, message, owner_id=None, job_id=None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "state": state,
        "message": message,
        "owner_id": owner_id,
        "job_id": job_id,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    with open(temporary, "w") as status_file:
        json.dump(status, status_file)
    os.replace(temporary, STATUS_PATH)


def read_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as metrics_file:
            return json.load(metrics_file)
    return {}


def run_retraining(owner_id, job_id, epochs=EPOCHS):
    """Run one account-scoped training job and persist its result or failure."""
    started_at = database.now()
    pending = database.get_pending_uploads(owner_id)
    job_state = "done"
    if not pending:
        message = "No new images to train on. Upload images first."
        write_status("idle", message, owner_id, job_id)
        database.finish_retraining(owner_id, job_id, "done")
        return

    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_renew_retraining_lease,
        args=(owner_id, job_id, stop_heartbeat),
        daemon=True,
    )
    heartbeat.start()
    try:
        class_names = config.load_class_names()
        uploads = database.get_all_uploads(owner_id)

        train_paths, train_labels = list_images(config.DATA_DIR / "train", class_names)
        val_paths, val_labels = list_images(config.DATA_DIR / "val", class_names)
        test_paths, test_labels = list_images(config.DATA_DIR / "test", class_names)
        train_paths.extend(upload["path"] for upload in uploads)
        train_labels.extend(class_names.index(upload["label"]) for upload in uploads)

        train_ds = make_dataset(train_paths, train_labels, shuffle=True)
        val_ds = make_dataset(val_paths, val_labels)
        test_ds = make_dataset(test_paths, test_labels)

        model = keras.models.load_model(config.MODEL_PATH, compile=False)
        compile_model(model, 1e-5)
        old_metrics = evaluate_model(model, test_ds, class_names)

        unfreeze_top_layers(model)
        compile_model(model, 1e-5)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=2,
        )

        new_metrics = evaluate_model(model, test_ds, class_names)
        promoted = new_metrics["accuracy"] >= old_metrics["accuracy"]
        if promoted:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            candidate_model = config.MODELS_DIR / "candidate.keras"
            candidate_h5 = config.MODELS_DIR / "candidate.h5"

            model.save(candidate_model)
            export_inference_model(model).save(candidate_h5)

            os.replace(candidate_model, config.MODEL_PATH)
            os.replace(candidate_h5, config.MODEL_H5_PATH)

            current_metrics = read_metrics()
            updated_metrics = dict(new_metrics)
            updated_metrics["baseline_accuracy"] = current_metrics.get("baseline_accuracy")
            updated_metrics["version"] = current_metrics.get("version", 1) + 1
            updated_metrics["note"] = f"Retrained with {len(uploads)} account-owned images"
            updated_metrics["trained_at"] = database.now()
            temporary_metrics = METRICS_PATH.with_suffix(".json.tmp")
            with open(temporary_metrics, "w") as metrics_file:
                json.dump(updated_metrics, metrics_file, indent=2)
            os.replace(temporary_metrics, METRICS_PATH)
            message = (
                f"New model saved: accuracy {old_metrics['accuracy']} "
                f"-> {new_metrics['accuracy']}"
            )
        else:
            message = (
                f"The new model was worse ({new_metrics['accuracy']} vs "
                f"{old_metrics['accuracy']}), so the previous model was kept."
            )

        database.mark_used([upload["id"] for upload in pending], owner_id)
        database.add_run(
            owner_id,
            started_at,
            len(pending),
            old_metrics["accuracy"],
            new_metrics["accuracy"],
            promoted,
            message,
            old_metrics,
            new_metrics,
        )
        write_status("done", message, owner_id, job_id)
    except Exception as error:
        job_state = "failed"
        message = f"Retraining failed: {error}"
        write_status("failed", message, owner_id, job_id)
        database.add_run(
            owner_id,
            started_at,
            len(pending),
            None,
            None,
            False,
            message,
        )
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join()
        database.finish_retraining(owner_id, job_id, job_state)
