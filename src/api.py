"""BeanGuard API for classification, authenticated retraining and dashboard data."""
import io
import os
import time
import uuid
import zipfile
import zlib
from datetime import datetime

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from src import auth, config, database, model_versions
from src.preprocessing import IMAGE_EXTENSIONS, validate_image
from src.prediction import predict_image
from src.retrain import (
    DEFAULT_EPOCHS,
    MAX_EPOCHS,
    read_status,
    run_retraining,
    write_status,
)

UI_DIST = config.BASE_DIR / "ui"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 100
MAX_ZIP_BYTES = 500 * 1024 * 1024
MAX_ZIP_FILES = 3000
MAX_ZIP_UNCOMPRESSED = 800 * 1024 * 1024

app = FastAPI(
    title="BeanGuard API",
    description=(
        "Classify bean-leaf photos as angular leaf spot, bean rust, or healthy. "
        "Use **POST /predict** with a JPEG or PNG image (multipart field `file`) for "
        "a prediction. Only bean-leaf images are meaningful; this is a screening aid, "
        "not a diagnosis. Account, upload, and retraining endpoints require a session "
        "created by **POST /auth/register** or **POST /auth/login**."
    ),
    openapi_tags=[
        {"name": "Health", "description": "Check whether the API is responding."},
        {"name": "Authentication", "description": "Create and manage your account session."},
        {"name": "Prediction", "description": "Classify one bean-leaf image."},
        {"name": "Models", "description": "Inspect status or choose an account model version."},
        {"name": "Training data", "description": "Upload labeled images for your account's retraining queue."},
        {"name": "Retraining", "description": "Start and monitor account-scoped model retraining."},
        {"name": "Insights", "description": "View dataset summaries for the dashboard."},
    ],
)
START_TIME = time.time()
database.init_db()


class RegisterPayload(BaseModel):
    email: str = Field(description="Email address used to sign in.")
    display_name: str = Field(description="Name shown in your BeanGuard account (1-80 characters).")
    password: str = Field(description="Password for this account (10-128 characters).")


class LoginPayload(BaseModel):
    email: str = Field(description="Email address used when registering.")
    password: str = Field(description="Password for this account.")


class ModelVersionPayload(BaseModel):
    version: str = Field(description="Available model version to use, such as v1.")


class RetrainPayload(BaseModel):
    epochs: int = Field(
        default=DEFAULT_EPOCHS,
        ge=1,
        le=MAX_EPOCHS,
        description=f"Maximum training epochs (1-{MAX_EPOCHS}); early stopping may finish sooner.",
    )


def _missing_zip_end_record(archive_bytes):
    end_record_start = max(0, len(archive_bytes) - (65_535 + 22))
    return (
        archive_bytes.startswith(b"PK\x03\x04")
        and archive_bytes.rfind(b"PK\x05\x06", end_record_start) == -1
    )


def _set_session(response, token):
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=auth.SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        path="/",
    )


def _optional_user(request):
    token = request.cookies.get(auth.SESSION_COOKIE)
    if not token:
        return None
    try:
        return auth.get_session_user(token)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _required_user(request):
    user = _optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Log in to upload data or retrain the model.")
    return user


def _new_session(response, user):
    try:
        token = auth.create_session(user["id"])
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    _set_session(response, token)


def _validate_auth_configuration():
    try:
        auth.validate_configuration()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def _read_bounded_upload(file, limit):
    contents = file.file.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than {limit // (1024 * 1024)} MB.")
    return contents


@app.get(
    "/health",
    summary="Check API availability",
    description="A lightweight liveness check. For model details, see `GET /status`.",
    tags=["Health"],
)
def health():
    return {"status": "ok"}


@app.post(
    "/auth/register",
    status_code=201,
    summary="Create an account",
    description="Provide an email, display name, and password. Creates an account and starts a session.",
    tags=["Authentication"],
)
def register(payload: RegisterPayload, response: Response):
    _validate_auth_configuration()
    try:
        user = auth.create_account(payload.email, payload.display_name, payload.password)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _new_session(response, user)
    return user


@app.post(
    "/auth/login",
    summary="Sign in",
    description="Provide the email and password used to register. Starts a session on success.",
    tags=["Authentication"],
)
def login(payload: LoginPayload, response: Response):
    _validate_auth_configuration()
    user = auth.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    _new_session(response, user)
    return user


@app.post(
    "/auth/logout",
    summary="Sign out",
    description="Ends the current session and clears its browser cookie.",
    tags=["Authentication"],
)
def logout(request: Request, response: Response):
    token = request.cookies.get(auth.SESSION_COOKIE)
    try:
        auth.delete_session(token)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    response.delete_cookie(
        auth.SESSION_COOKIE,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        path="/",
    )
    return {"message": "Logged out."}


@app.get(
    "/auth/me",
    summary="Get the signed-in account",
    description="Returns the current account when a valid session cookie is present.",
    tags=["Authentication"],
)
def current_user(request: Request):
    user = _optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


@app.get(
    "/status",
    summary="Get service and model status",
    description="Returns model versions and evaluation metrics; upload counts and retraining history are scoped to the signed-in account.",
    tags=["Models"],
)
def status(request: Request):
    """Return public model status and account-scoped upload and retraining details."""
    user = _optional_user(request)
    retrain = read_status()
    if retrain.get("state") == "running" and not database.retraining_active():
        message = "Retraining stopped before it finished. Start retraining again."
        write_status("failed", message, retrain.get("owner_id"), retrain.get("job_id"))
        retrain = read_status()
    if not user or retrain.get("owner_id") != user["id"]:
        retrain = {
            "state": retrain["state"],
            "message": "A retraining run is in progress." if retrain["state"] == "running" else "",
            "updated_at": retrain.get("updated_at"),
        }

    active_version = (
        user["active_model_version"] if user else model_versions.INITIAL_VERSION
    )
    versions = model_versions.available_versions(active_version)
    active_metrics = model_versions.read_metrics(active_version)
    return {
        "status": "online",
        "uptime_seconds": int(time.time() - START_TIME),
        "started_at": datetime.fromtimestamp(START_TIME).isoformat(timespec="seconds"),
        "model_available": True,
        "active_model": active_version,
        "model_versions": versions,
        "metrics": active_metrics,
        "classes": config.load_class_names(),
        "retrain": retrain,
        "pending_uploads": database.count_pending(user["id"]) if user else None,
        "total_uploads": database.count_all_uploads(user["id"]) if user else None,
        "recent_runs": database.recent_runs(user["id"]) if user else [],
        "user": user,
        "retrain_defaults": {
            "epochs": DEFAULT_EPOCHS,
            "max_epochs": MAX_EPOCHS,
        },
    }


@app.post(
    "/model-version",
    summary="Choose a model version",
    description="Select an available model for predictions made by your account. This does not change other accounts' selections.",
    tags=["Models"],
)
def select_model_version(payload: ModelVersionPayload, request: Request):
    """Select the prediction model for the signed-in account only."""
    user = _required_user(request)
    try:
        model_versions.model_path(payload.version)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    database.set_active_model_version(user["id"], payload.version)
    return {"active_model": payload.version}


@app.post(
    "/predict",
    summary="Predict a bean-leaf class",
    description=(
        "Upload one JPEG or PNG bean-leaf photo as multipart field `file` (maximum 10 MB). "
        "Returns the predicted class, confidence, class probabilities, and model version. "
        "Login is optional; signed-in users use their selected model, while other requests use v1. "
        "Other kinds of images may still receive a confident but meaningless result."
    ),
    tags=["Prediction"],
)
async def predict(
    request: Request,
    file: UploadFile = File(
        ...,
        description="One JPEG or PNG bean-leaf photo, maximum 10 MB.",
    ),
):
    """Classify one valid bean-leaf image."""
    contents = _read_bounded_upload(file, MAX_UPLOAD_BYTES)
    user = _optional_user(request) if request else None
    version = (
        user["active_model_version"] if user else model_versions.INITIAL_VERSION
    )
    try:
        result = await run_in_threadpool(predict_image, contents, version)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return result


def store_image(image, label, owner_id):
    """Save an image and record its owner for isolated future training runs."""
    folder = config.UPLOADS_DIR / label
    path = folder / f"{uuid.uuid4().hex}.jpg"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        image.save(path, "JPEG", quality=95)
    except PermissionError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The server cannot write to upload storage. Give the API process "
                "write access to data/uploads, then retry the upload."
            ),
        ) from error
    database.add_upload(path, label, owner_id)


def schedule_retraining(background_tasks, owner_id, message, epochs, starting_version):
    job_id = database.claim_retraining(owner_id)
    if job_id is None:
        return False
    try:
        write_status(
            "running", message, owner_id, job_id,
            stage="queued", epoch=0, epochs=epochs, logs={},
        )
        background_tasks.add_task(
            run_retraining,
            owner_id,
            job_id,
            epochs=epochs,
            starting_version=starting_version,
        )
    except Exception:
        database.finish_retraining(owner_id, job_id, "failed")
        raise
    return True


@app.post(
    "/upload",
    summary="Upload labeled images",
    description=(
        "Requires a signed-in account. Upload 1–100 JPEG or PNG images and choose the correct "
        "class label for all files. Each image is limited to 10 MB. Accepted images are queued "
        "for your next retraining run; they do not immediately alter the model."
    ),
    tags=["Training data"],
)
async def upload(
    request: Request,
    label: str = Form(
        ...,
        description="Correct class label for every image: angular_leaf_spot, bean_rust, or healthy.",
    ),
    files: list[UploadFile] = File(
        ...,
        description="Between 1 and 100 JPEG or PNG images, each no larger than 10 MB.",
    ),
):
    """Queue labeled images for retraining; this does not immediately change the model."""
    user = _required_user(request)
    class_names = config.load_class_names()
    if label not in class_names:
        raise HTTPException(status_code=400, detail=f"label must be one of {class_names}")
    if not files or len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"Upload between 1 and {MAX_BATCH_FILES} images.")

    saved, rejected = 0, []
    for file in files:
        try:
            contents = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{file.filename} exceeds the 10 MB limit.")
            image = validate_image(contents)
            store_image(image, label, user["id"])
            saved += 1
        except ValueError:
            rejected.append(file.filename or "unnamed file")

    if not saved:
        raise HTTPException(status_code=400, detail="No valid JPEG or PNG images were uploaded.")
    return {
        "saved": saved,
        "rejected": rejected,
        "pending_uploads": database.count_pending(user["id"]),
    }


def normalise(name):
    return name.strip().lower().replace(" ", "_").replace("-", "_")


@app.post(
    "/upload-zip",
    summary="Upload a ZIP of labeled images",
    description=(
        "Requires a signed-in account. Upload one standard ZIP (up to 500 MB) with class-named "
        "folders at its root, for example `healthy/leaf.jpg` and `bean_rust/leaf.jpg`. "
        "Images are queued for your next retraining run; they do not immediately alter the model."
    ),
    tags=["Training data"],
)
async def upload_zip(
    request: Request,
    file: UploadFile = File(
        ...,
        description="A single ZIP archive with angular_leaf_spot/, bean_rust/, and/or healthy/ folders at its root.",
    ),
):
    """Upload a zip organized as one folder per class label."""
    user = _required_user(request)
    class_names = config.load_class_names()
    archive_bytes = await file.read(MAX_ZIP_BYTES + 1)
    if len(archive_bytes) > MAX_ZIP_BYTES:
        limit_mb = MAX_ZIP_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Zip file exceeds the {limit_mb} MB limit.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except (zipfile.BadZipFile, EOFError, OSError) as error:
        size_mb = len(archive_bytes) / (1024 * 1024)
        if _missing_zip_end_record(archive_bytes):
            detail = (
                f"The uploaded file is {size_mb:.1f} MB and starts like a ZIP, but its "
                "end-of-central-directory record is missing. The archive is incomplete "
                "or split across multiple files; recreate it as a standard single-file "
                "ZIP and upload that complete file."
            )
        else:
            detail = (
                f"Could not open the uploaded file as a ZIP archive ({size_mb:.1f} MB). "
                f"ZIP parser reason: {error}. The file may be incomplete or damaged, renamed "
                "from another format, or part of a multi-volume archive. Create a standard "
                "single-file .zip and retry."
            )
        raise HTTPException(status_code=400, detail=detail) from error

    with archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if (
            len(members) > MAX_ZIP_FILES
            or sum(member.file_size for member in members) > MAX_ZIP_UNCOMPRESSED
        ):
            raise HTTPException(status_code=400, detail="Zip exceeds the unpacked file limits.")

        per_class = {class_name: 0 for class_name in class_names}
        unknown_folders, broken = set(), 0
        for member in members:
            parts = member.filename.replace("\\", "/").split("/")
            filename = parts[-1]
            if "__MACOSX" in parts or filename.startswith(".") or not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            folder = normalise(parts[-2]) if len(parts) > 1 else ""
            if folder not in class_names:
                unknown_folders.add(parts[-2] if len(parts) > 1 else "(top level)")
                continue
            if member.file_size > MAX_UPLOAD_BYTES:
                broken += 1
                continue
            try:
                contents = archive.read(member)
                image = validate_image(contents)
            except (ValueError, zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, OSError, zlib.error):
                broken += 1
                continue
            store_image(image, folder, user["id"])
            per_class[folder] += 1

    total = sum(per_class.values())
    if total == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No usable images found. Zip folders must use these class names: {class_names}.",
        )

    return {
        "saved": total,
        "per_class": per_class,
        "broken_files": broken,
        "unknown_folders": sorted(unknown_folders),
        "pending_uploads": database.count_pending(user["id"]),
    }


@app.post(
    "/retrain",
    summary="Start model retraining",
    description=(
        "Requires a signed-in account with pending uploaded images. Optionally provide JSON "
        "`{\"epochs\": 5}`; the range is 1–20 and the default is 5. Training runs in the background "
        "from your selected model. Check `GET /status` for progress and outcome."
    ),
    tags=["Retraining"],
)
def retrain(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: RetrainPayload = RetrainPayload(),
):
    """Start a training run using only this account's uploads."""
    user = _required_user(request)
    if database.count_pending(user["id"]) == 0:
        raise HTTPException(status_code=400, detail="Upload new images before retraining.")
    if not schedule_retraining(
        background_tasks,
        user["id"],
        "Retraining started.",
        payload.epochs,
        user["active_model_version"],
    ):
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    return {"message": "Retraining started.", "epochs": payload.epochs}


_colour_cache = {}


def _count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for filename in os.listdir(folder) if filename.lower().endswith(IMAGE_EXTENSIONS))


@app.get(
    "/insights",
    summary="Get dataset insights",
    description="Returns image counts by dataset split and class, plus average RGB values for dashboard exploration.",
    tags=["Insights"],
)
def insights():
    """Dataset split counts and average RGB values for visual exploration."""
    class_names = config.load_class_names()
    counts = {
        split: {
            class_name: _count_images(config.DATA_DIR / split / class_name)
            for class_name in class_names
        }
        for split in ("train", "val", "test")
    }

    if not _colour_cache:
        for class_name in class_names:
            folder = config.DATA_DIR / "train" / class_name
            files = sorted(os.listdir(folder))[:60] if os.path.isdir(folder) else []
            means = [
                np.asarray(Image.open(folder / filename).convert("RGB").resize((64, 64))).mean(axis=(0, 1))
                for filename in files
                if filename.lower().endswith(IMAGE_EXTENSIONS)
            ]
            if means:
                red, green, blue = np.mean(means, axis=0)
                _colour_cache[class_name] = {
                    "red": round(float(red), 1),
                    "green": round(float(green), 1),
                    "blue": round(float(blue), 1),
                }
    return {"classes": class_names, "counts": counts, "colours": _colour_cache}


if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        tags=app.openapi_tags,
        routes=app.routes,
    )

    def mark_binary_uploads(node):
        if isinstance(node, dict):
            if (
                node.get("type") == "string"
                and node.get("contentMediaType") == "application/octet-stream"
            ):
                node.pop("contentMediaType")
                node["format"] = "binary"
            for value in node.values():
                mark_binary_uploads(value)
        elif isinstance(node, list):
            for value in node:
                mark_binary_uploads(value)

    mark_binary_uploads(schema)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
