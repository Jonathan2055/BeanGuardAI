"""Immutable shared model artifacts with account-scoped active version selection."""
import json
import os
import re
import shutil

from src import config

INITIAL_VERSION = "v1"
VERSION_DIR = config.MODELS_DIR / "versions"
_VERSION_PATTERN = re.compile(r"v([1-9][0-9]*)\Z")


def _version_number(version):
    match = _VERSION_PATTERN.fullmatch(version)
    if not match:
        raise ValueError(f"Unknown model version: {version}")
    return int(match.group(1))


def _paths(version):
    _version_number(version)
    return (
        VERSION_DIR / f"{version}.keras",
        VERSION_DIR / f"{version}.h5",
        VERSION_DIR / f"{version}.json",
    )


def ensure_initial_version():
    """Snapshot the notebook model once so later promoted versions remain immutable."""
    model, inference_model, metadata = _paths(INITIAL_VERSION)
    if metadata.is_file() and model.is_file():
        return
    if not config.MODEL_PATH.is_file():
        raise FileNotFoundError("The initial model is missing. Train and save it first.")

    VERSION_DIR.mkdir(parents=True, exist_ok=True)
    if not model.is_file():
        temporary_model = model.with_name(f"{model.stem}.tmp{model.suffix}")
        shutil.copy2(config.MODEL_PATH, temporary_model)
        os.replace(temporary_model, model)
    if config.MODEL_H5_PATH.is_file() and not inference_model.is_file():
        temporary_inference = inference_model.with_name(
            f"{inference_model.stem}.tmp{inference_model.suffix}"
        )
        shutil.copy2(config.MODEL_H5_PATH, temporary_inference)
        os.replace(temporary_inference, inference_model)

    metrics = {}
    if config.MODELS_DIR.joinpath("metrics.json").is_file():
        with open(config.MODELS_DIR / "metrics.json") as metrics_file:
            metrics = json.load(metrics_file)
    metrics.setdefault("version", 1)
    metrics.setdefault("evaluation_split", "test")
    temporary = metadata.with_suffix(".json.tmp")
    with open(temporary, "w") as metadata_file:
        json.dump(metrics, metadata_file, indent=2)
    os.replace(temporary, metadata)


def model_path(version):
    ensure_initial_version()
    model, _, metadata = _paths(version)
    if not model.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"The {version} model is not available.")
    return model


def read_metrics(version):
    ensure_initial_version()
    _, _, metadata = _paths(version)
    if not metadata.is_file():
        raise FileNotFoundError(f"The {version} model is not available.")
    with open(metadata) as metadata_file:
        return json.load(metadata_file)


def save_version(model_source, inference_source, metrics):
    """Atomically register model artifacts and metadata under the next immutable ID."""
    ensure_initial_version()
    version_ids = []
    for path in VERSION_DIR.glob("v*.keras"):
        match = _VERSION_PATTERN.fullmatch(path.stem)
        if match:
            version_ids.append(int(match.group(1)))
    version = f"v{max(version_ids, default=0) + 1}"
    model, inference_model, metadata = _paths(version)
    VERSION_DIR.mkdir(parents=True, exist_ok=True)

    for source, destination in ((model_source, model), (inference_source, inference_model)):
        temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    version_metrics = dict(metrics)
    version_metrics["version"] = _version_number(version)
    temporary_metadata = metadata.with_suffix(".json.tmp")
    with open(temporary_metadata, "w") as metadata_file:
        json.dump(version_metrics, metadata_file, indent=2)
    os.replace(temporary_metadata, metadata)
    return version


def available_versions(active_version=INITIAL_VERSION):
    ensure_initial_version()
    versions = []
    for metadata in VERSION_DIR.glob("v*.json"):
        version = metadata.stem
        if not _VERSION_PATTERN.fullmatch(version):
            continue
        model = VERSION_DIR / f"{version}.keras"
        if not model.is_file():
            continue
        versions.append({
            "id": version,
            "metrics": read_metrics(version),
            "active": version == active_version,
        })
    return sorted(versions, key=lambda version: _version_number(version["id"]))
