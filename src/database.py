"""SQLite storage for accounts, sessions, uploads and retraining runs."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

from src import config

DB_PATH = config.DATA_DIR / "app.db"
RETRAINING_LEASE = timedelta(minutes=2)


@contextmanager
def connect():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _columns(db, table):
    return {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}


def init_db():
    """Create the schema and add ownership fields to databases from earlier versions."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_digest TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                label TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS retrain_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                finished_at TEXT,
                new_images INTEGER,
                old_accuracy REAL,
                new_accuracy REAL,
                promoted INTEGER,
                message TEXT
            );
            CREATE TABLE IF NOT EXISTS retrain_jobs (
                singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                job_id TEXT NOT NULL,
                owner_id INTEGER NOT NULL REFERENCES users(id),
                state TEXT NOT NULL,
                claimed_at TEXT NOT NULL
            );
        """)

        upload_columns = _columns(db, "uploads")
        if "owner_id" not in upload_columns:
            db.execute("ALTER TABLE uploads ADD COLUMN owner_id INTEGER REFERENCES users(id)")

        run_columns = _columns(db, "retrain_runs")
        if "owner_id" not in run_columns:
            db.execute("ALTER TABLE retrain_runs ADD COLUMN owner_id INTEGER REFERENCES users(id)")
        if "old_metrics" not in run_columns:
            db.execute("ALTER TABLE retrain_runs ADD COLUMN old_metrics TEXT")
        if "new_metrics" not in run_columns:
            db.execute("ALTER TABLE retrain_runs ADD COLUMN new_metrics TEXT")

        db.execute("CREATE INDEX IF NOT EXISTS uploads_owner_pending ON uploads(owner_id, used)")
        db.execute("CREATE INDEX IF NOT EXISTS retrain_runs_owner ON retrain_runs(owner_id, id)")
        db.execute("CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at)")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_user(email, display_name, password_hash):
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO users (email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (email, display_name, password_hash, now()),
        )
        return cursor.lastrowid


def user_by_email(email):
    with connect() as db:
        row = db.execute(
            "SELECT id, email, display_name, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    return dict(row) if row else None


def user_by_id(user_id):
    with connect() as db:
        row = db.execute(
            "SELECT id, email, display_name FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def add_session(token_digest, user_id, expires_at):
    with connect() as db:
        db.execute(
            "INSERT INTO sessions (token_digest, user_id, expires_at) VALUES (?, ?, ?)",
            (token_digest, user_id, expires_at),
        )


def session_user(token_digest, current_time):
    with connect() as db:
        row = db.execute(
            """SELECT users.id, users.email, users.display_name
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token_digest = ? AND sessions.expires_at > ?""",
            (token_digest, current_time),
        ).fetchone()
    return dict(row) if row else None


def remove_session(token_digest):
    with connect() as db:
        db.execute("DELETE FROM sessions WHERE token_digest = ?", (token_digest,))


def add_upload(path, label, owner_id):
    with connect() as db:
        db.execute(
            "INSERT INTO uploads (path, label, uploaded_at, owner_id) VALUES (?, ?, ?, ?)",
            (str(path), label, now(), owner_id),
        )


def get_all_uploads(owner_id):
    with connect() as db:
        rows = db.execute(
            "SELECT id, path, label, owner_id FROM uploads WHERE owner_id = ? ORDER BY id",
            (owner_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_pending_uploads(owner_id):
    with connect() as db:
        rows = db.execute(
            """SELECT id, path, label, owner_id FROM uploads
               WHERE owner_id = ? AND used = 0 ORDER BY id""",
            (owner_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def count_pending(owner_id):
    with connect() as db:
        return db.execute(
            "SELECT COUNT(*) FROM uploads WHERE owner_id = ? AND used = 0",
            (owner_id,),
        ).fetchone()[0]


def count_all_uploads(owner_id):
    with connect() as db:
        return db.execute(
            "SELECT COUNT(*) FROM uploads WHERE owner_id = ?", (owner_id,)
        ).fetchone()[0]


def mark_used(upload_ids, owner_id):
    if not upload_ids:
        return
    with connect() as db:
        db.executemany(
            "UPDATE uploads SET used = 1 WHERE id = ? AND owner_id = ?",
            [(upload_id, owner_id) for upload_id in upload_ids],
        )


def add_run(
    owner_id,
    started_at,
    new_images,
    old_accuracy,
    new_accuracy,
    promoted,
    message,
    old_metrics=None,
    new_metrics=None,
):
    with connect() as db:
        db.execute(
            """INSERT INTO retrain_runs
               (owner_id, started_at, finished_at, new_images, old_accuracy,
                new_accuracy, promoted, message, old_metrics, new_metrics)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                owner_id,
                started_at,
                now(),
                new_images,
                old_accuracy,
                new_accuracy,
                int(promoted),
                message,
                json.dumps(old_metrics) if old_metrics is not None else None,
                json.dumps(new_metrics) if new_metrics is not None else None,
            ),
        )


def recent_runs(owner_id, limit=5):
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM retrain_runs WHERE owner_id = ? ORDER BY id DESC LIMIT ?",
            (owner_id, limit),
        ).fetchall()
    runs = []
    for row in rows:
        run = dict(row)
        for field in ("old_metrics", "new_metrics"):
            run[field] = json.loads(run[field]) if run[field] else None
        runs.append(run)
    return runs


def claim_retraining(owner_id):
    """Atomically claim the single shared-model training slot across API replicas."""
    job_id = uuid.uuid4().hex
    current_time = datetime.now().astimezone()
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        active = db.execute(
            "SELECT claimed_at FROM retrain_jobs WHERE singleton_id = 1 AND state = 'running'"
        ).fetchone()
        if active:
            claimed_at = datetime.fromisoformat(active["claimed_at"])
            if current_time - claimed_at < RETRAINING_LEASE:
                return None
            db.execute("DELETE FROM retrain_jobs WHERE singleton_id = 1")
        db.execute(
            """INSERT INTO retrain_jobs (singleton_id, job_id, owner_id, state, claimed_at)
               VALUES (1, ?, ?, 'running', ?)""",
            (job_id, owner_id, current_time.isoformat(timespec="seconds")),
        )
    return job_id


def finish_retraining(owner_id, job_id, state):
    with connect() as db:
        db.execute(
            """UPDATE retrain_jobs SET state = ?
               WHERE singleton_id = 1 AND owner_id = ? AND job_id = ? AND state = 'running'""",
            (state, owner_id, job_id),
        )


def renew_retraining(owner_id, job_id):
    with connect() as db:
        cursor = db.execute(
            """UPDATE retrain_jobs SET claimed_at = ?
               WHERE singleton_id = 1 AND owner_id = ? AND job_id = ? AND state = 'running'""",
            (now(), owner_id, job_id),
        )
    return cursor.rowcount == 1


def retraining_active():
    current_time = datetime.now().astimezone()
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            """SELECT job_id, claimed_at FROM retrain_jobs
               WHERE singleton_id = 1 AND state = 'running'"""
        ).fetchone()
        if not row:
            return False
        if current_time - datetime.fromisoformat(row["claimed_at"]) >= RETRAINING_LEASE:
            db.execute(
                "DELETE FROM retrain_jobs WHERE singleton_id = 1 AND job_id = ?",
                (row["job_id"],),
            )
            return False
        return True
