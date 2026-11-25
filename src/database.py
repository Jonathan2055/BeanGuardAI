"""Database (SQLite): remembers every uploaded image and every retraining run."""
import sqlite3
from datetime import datetime

from src import config

DB_PATH = config.DATA_DIR / "app.db"


def connect():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row          # lets us read columns by name
    return connection


def init_db():
    """Create the two tables if they do not exist yet."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            label TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0)""")
        db.execute("""CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT, finished_at TEXT, new_images INTEGER,
            old_accuracy REAL, new_accuracy REAL, promoted INTEGER, message TEXT)""")


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---- uploaded images ----
def add_upload(path, label):
    with connect() as db:
        db.execute("INSERT INTO uploads (path, label, uploaded_at) VALUES (?, ?, ?)",
                   (str(path), label, now()))


def get_all_uploads():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT id, path, label FROM uploads")]


def get_pending_uploads():
    """Images uploaded but not used in a retraining run yet."""
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT id, path, label FROM uploads WHERE used = 0")]


def count_pending():
    with connect() as db:
        return db.execute("SELECT COUNT(*) FROM uploads WHERE used = 0").fetchone()[0]


def count_all_uploads():
    with connect() as db:
        return db.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]


def mark_used(upload_ids):
    with connect() as db:
        db.executemany("UPDATE uploads SET used = 1 WHERE id = ?", [(i,) for i in upload_ids])


# ---- retraining history ----
def add_run(started_at, new_images, old_accuracy, new_accuracy, promoted, message):
    with connect() as db:
        db.execute("""INSERT INTO retrain_runs
            (started_at, finished_at, new_images, old_accuracy, new_accuracy, promoted, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                   (started_at, now(), new_images, old_accuracy, new_accuracy, int(promoted), message))


def recent_runs(limit=5):
    with connect() as db:
        rows = db.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
