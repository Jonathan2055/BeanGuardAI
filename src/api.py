"""BeanGuard API for classification, authenticated retraining and dashboard data."""
import io
import json
import os
import time
import uuid
import zipfile
from datetime import datetime

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src import auth, config, database, model_versions
from src.preprocessing import IMAGE_EXTENSIONS, validate_image
from src.prediction import predict_image
from src.retrain import METRICS_PATH, read_status, run_retraining, write_status

UI_DIST = config.BASE_DIR / "ui"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 100
MAX_ZIP_BYTES = 250 * 1024 * 1024
MAX_ZIP_FILES = 3000
MAX_ZIP_UNCOMPRESSED = 600 * 1024 * 1024

app = FastAPI(title="BeanGuard API")
START_TIME = time.time()
database.init_db()


class RegisterPayload(BaseModel):
    email: str
    display_name: str
    password: str


class LoginPayload(BaseModel):
    email: str
    password: str


class ModelVersionPayload(BaseModel):
    version: str


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(payload: RegisterPayload, response: Response):
    _validate_auth_configuration()
    try:
        user = auth.create_account(payload.email, payload.display_name, payload.password)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _new_session(response, user)
    return user


@app.post("/auth/login")
def login(payload: LoginPayload, response: Response):
    _validate_auth_configuration()
    user = auth.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    _new_session(response, user)
    return user


@app.post("/auth/logout")
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


@app.get("/auth/me")
def current_user(request: Request):
    user = _optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


@app.get("/status")
def status(request: Request):
    """Public model status with private upload counts and history scoped to the logged-in user."""
    user = _optional_user(request)
    metrics = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

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

    active_version = model_versions.get_active_version()
    versions = model_versions.available_versions()
    return {
        "status": "online",
        "uptime_seconds": int(time.time() - START_TIME),
        "started_at": datetime.fromtimestamp(START_TIME).isoformat(timespec="seconds"),
        "model_available": model_versions.model_path(active_version).exists(),
        "active_model": active_version,
        "model_versions": versions,
        "metrics": model_versions.read_metrics(active_version) if versions else metrics,
        "classes": config.load_class_names(),
        "retrain": retrain,
        "pending_uploads": database.count_pending(user["id"]) if user else None,
        "total_uploads": database.count_all_uploads(user["id"]) if user else None,
        "recent_runs": database.recent_runs(user["id"]) if user else [],
        "user": user,
    }


@app.post("/model-version")
def select_model_version(payload: ModelVersionPayload, request: Request):
    """Select the model used by predictions for all signed-in users."""
    _required_user(request)
    try:
        model_versions.set_active_version(payload.version)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"active_model": payload.version}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify one valid image; users should upload bean-leaf photos for meaningful results."""
    contents = _read_bounded_upload(file, MAX_UPLOAD_BYTES)
    try:
        result = await run_in_threadpool(predict_image, contents)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return result


def store_image(image, label, owner_id):
    """Save an image and record its owner for isolated future training runs."""
    folder = config.UPLOADS_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.jpg"
    image.save(path, "JPEG", quality=95)
    database.add_upload(path, label, owner_id)


def schedule_retraining(background_tasks, owner_id, message):
    job_id = database.claim_retraining(owner_id)
    if job_id is None:
        return False
    try:
        write_status("running", message, owner_id, job_id)
        background_tasks.add_task(run_retraining, owner_id, job_id)
    except Exception:
        database.finish_retraining(owner_id, job_id, "failed")
        raise
    return True


@app.post("/upload")
async def upload(
    request: Request,
    label: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload multiple labeled images for the authenticated user's retraining data."""
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


@app.post("/upload-zip")
async def upload_zip(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a zip organized as one folder per class label."""
    user = _required_user(request)
    class_names = config.load_class_names()
    archive_bytes = await file.read(MAX_ZIP_BYTES + 1)
    if len(archive_bytes) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="Zip file exceeds the 250 MB limit.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as error:
        raise HTTPException(status_code=400, detail="This is not a valid .zip file.") from error

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
            store_image(image, folder, user["id"])
            per_class[folder] += 1
        except ValueError:
            broken += 1

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


@app.post("/retrain")
def retrain(request: Request, background_tasks: BackgroundTasks):
    """Start a training run using only this account's uploads."""
    user = _required_user(request)
    if database.count_pending(user["id"]) == 0:
        raise HTTPException(status_code=400, detail="Upload new images before retraining.")
    if not schedule_retraining(background_tasks, user["id"], "Retraining started."):
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    return {"message": "Retraining started."}


_colour_cache = {}


def _count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for filename in os.listdir(folder) if filename.lower().endswith(IMAGE_EXTENSIONS))


@app.get("/insights")
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
