"""Model files and the globally active prediction version."""
import json
import os
import shutil

from src import config

CURRENT = "current"
PREVIOUS = "previous"

MODEL_PATHS = {
    CURRENT: config.MODEL_PATH,
    PREVIOUS: config.MODELS_DIR / "bean_model_previous.keras",
}
METRICS_PATHS = {
    CURRENT: config.MODELS_DIR / "metrics.json",
    PREVIOUS: config.MODELS_DIR / "metrics_previous.json",
}
INFERENCE_MODEL_PATHS = {
    CURRENT: config.MODEL_H5_PATH,
    PREVIOUS: config.MODELS_DIR / "bean_model_previous.h5",
}
ACTIVE_MODEL_PATH = config.MODELS_DIR / "active_model.json"


def model_path(version):
    if version not in MODEL_PATHS:
        raise ValueError(f"Unknown model version: {version}")
    return MODEL_PATHS[version]


def get_active_version():
    if not ACTIVE_MODEL_PATH.exists():
        return CURRENT
    with open(ACTIVE_MODEL_PATH) as active_file:
        active_version = json.load(active_file).get("version")
    if active_version not in MODEL_PATHS:
        raise ValueError("The active model version setting is invalid.")
    return active_version


def set_active_version(version):
    path = model_path(version)
    if not path.is_file():
        raise FileNotFoundError(f"The {version} model is not available.")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = ACTIVE_MODEL_PATH.with_suffix(".json.tmp")
    with open(temporary_path, "w") as active_file:
        json.dump({"version": version}, active_file)
    os.replace(temporary_path, ACTIVE_MODEL_PATH)


def read_metrics(version):
    metrics_path = METRICS_PATHS[version]
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as metrics_file:
        return json.load(metrics_file)


def preserve_current_version():
    """Snapshot the current model artifacts and their metrics as the previous version."""
    current_metrics = read_metrics(CURRENT)
    for version_paths in (MODEL_PATHS, INFERENCE_MODEL_PATHS):
        source = version_paths[CURRENT]
        destination = version_paths[PREVIOUS]
        if source.is_file():
            temporary_path = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
            shutil.copy2(source, temporary_path)
            os.replace(temporary_path, destination)
    metrics_path = METRICS_PATHS[PREVIOUS]
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    with open(temporary_metrics, "w") as metrics_file:
        json.dump(current_metrics, metrics_file, indent=2)
    os.replace(temporary_metrics, metrics_path)
    return current_metrics


def available_versions():
    active_version = get_active_version()
    versions = []
    for version, path in MODEL_PATHS.items():
        if path.is_file():
            versions.append({
                "id": version,
                "metrics": read_metrics(version),
                "active": version == active_version,
            })
    return versions
