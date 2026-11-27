"""Fine-tune a user's selected model and publish promoted versions immutably."""
import json
import logging
import os
import threading
from datetime import datetime

from tensorflow import keras

from src import config, database, model_versions
from src.model import compile_model, evaluate_model, export_inference_model, unfreeze_top_layers
from src.preprocessing import list_images, make_dataset

STATUS_PATH = config.MODELS_DIR / "retrain_status.json"
DEFAULT_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))
MAX_EPOCHS = 20
if not 1 <= DEFAULT_EPOCHS <= MAX_EPOCHS:
    raise ValueError(f"RETRAIN_EPOCHS must be between 1 and {MAX_EPOCHS}.")
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


def write_status(state, message, owner_id=None, job_id=None, **details):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "state": state,
        "message": message,
        "owner_id": owner_id,
        "job_id": job_id,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **details,
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    with open(temporary, "w") as status_file:
        json.dump(status, status_file)
    os.replace(temporary, STATUS_PATH)


class EpochProgress(keras.callbacks.Callback):
    def __init__(self, owner_id, job_id, epochs):
        super().__init__()
        self.owner_id = owner_id
        self.job_id = job_id
        self.epochs = epochs
        self.completed_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        self.completed_epochs = epoch + 1
        logs = logs or {}
        values = {
            key: round(float(value), 4)
            for key, value in logs.items()
            if value is not None
        }
        write_status(
            "running",
            f"Completed epoch {epoch + 1} of {self.epochs}.",
            self.owner_id,
            self.job_id,
            stage="training",
            epoch=epoch + 1,
            epochs=self.epochs,
            logs=values,
        )


def run_retraining(owner_id, job_id, epochs=DEFAULT_EPOCHS, starting_version=None):
    """Train from the requesting account's selected model and promote on validation gains."""
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
    candidate_model = config.MODELS_DIR / f"candidate-{job_id}.keras"
    candidate_h5 = config.MODELS_DIR / f"candidate-{job_id}.h5"
    try:
        write_status(
            "running", "Preparing training data.", owner_id, job_id,
            stage="preparing", epoch=0, epochs=epochs, logs={},
        )
        class_names = config.load_class_names()
        uploads = database.get_all_uploads(owner_id)
        user = database.user_by_id(owner_id)
        if not user:
            raise ValueError("The account associated with this training run no longer exists.")
        starting_version = starting_version or user["active_model_version"]

        train_paths, train_labels = list_images(config.DATA_DIR / "train", class_names)
        val_paths, val_labels = list_images(config.DATA_DIR / "val", class_names)
        train_paths.extend(upload["path"] for upload in uploads)
        train_labels.extend(class_names.index(upload["label"]) for upload in uploads)

        train_ds = make_dataset(train_paths, train_labels, shuffle=True)
        val_ds = make_dataset(val_paths, val_labels)

        model = keras.models.load_model(
            model_versions.model_path(starting_version), compile=False
        )
        compile_model(model, 1e-5)
        old_metrics = evaluate_model(model, val_ds, class_names)

        write_status(
            "running", "Fine-tuning the selected model.", owner_id, job_id,
            stage="training", epoch=0, epochs=epochs, logs={},
        )
        unfreeze_top_layers(model)
        compile_model(model, 1e-5)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
        progress_callback = EpochProgress(owner_id, job_id, epochs)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[
                early_stopping,
                progress_callback,
            ],
            verbose=2,
        )

        write_status(
            "running", "Evaluating the candidate on the validation split.",
            owner_id, job_id, stage="evaluating",
            epoch=progress_callback.completed_epochs, epochs=epochs, logs={},
        )
        new_metrics = evaluate_model(model, val_ds, class_names)
        promoted = new_metrics["accuracy"] > old_metrics["accuracy"]
        new_version = None
        if promoted:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.save(candidate_model)
            export_inference_model(model).save(candidate_h5)

            version_metrics = dict(new_metrics)
            version_metrics.update({
                "evaluation_split": "validation",
                "trained_at": database.now(),
                "trained_epochs": epochs,
                "improved_from_version": starting_version,
            })
            new_version = model_versions.save_version(
                candidate_model, candidate_h5, version_metrics
            )
            still_selected = database.user_by_id(owner_id)["active_model_version"] == starting_version
            if still_selected:
                database.set_active_model_version(owner_id, new_version)
            message = (
                f"Promoted {new_version}: validation accuracy improved from "
                f"{old_metrics['accuracy']} to {new_metrics['accuracy']}."
            )
            if not still_selected:
                message += " Your newer model selection was kept."
        else:
            message = (
                f"No promotion: validation accuracy {new_metrics['accuracy']} did not "
                f"exceed {starting_version} ({old_metrics['accuracy']})."
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
        write_status(
            "done", message, owner_id, job_id,
            stage="complete", epoch=progress_callback.completed_epochs,
            epochs=epochs, logs={},
            promoted=promoted,
            model_version=database.user_by_id(owner_id)["active_model_version"],
        )
    except Exception as error:
        job_state = "failed"
        message = f"Retraining failed: {error}"
        write_status(
            "failed", message, owner_id, job_id,
            stage="failed", epoch=0, epochs=epochs, logs={},
        )
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
        for candidate in (candidate_model, candidate_h5):
            if candidate.exists():
                candidate.unlink()
        stop_heartbeat.set()
        heartbeat.join()
        database.finish_retraining(owner_id, job_id, job_state)
