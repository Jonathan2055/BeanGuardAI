import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from unittest.mock import patch

from src import auth, database


class AccountAndOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "app.db"
        self.db_path_patch = patch.object(database, "DB_PATH", self.database_path)
        self.db_path_patch.start()
        self.secret_patch = patch.dict(os.environ, {"APP_SECRET_KEY": "test-key-" + "x" * 40})
        self.secret_patch.start()
        database.init_db()
        self.alice = auth.create_account("alice@example.test", "Alice", "correct-horse-battery")
        self.bob = auth.create_account("bob@example.test", "Bob", "another-correct-password")

    def tearDown(self):
        self.secret_patch.stop()
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_password_hash_and_session_lifecycle(self):
        stored = database.user_by_email("alice@example.test")
        self.assertNotIn("correct-horse-battery", stored["password_hash"])
        self.assertTrue(auth.verify_password("correct-horse-battery", stored["password_hash"]))
        self.assertFalse(auth.verify_password("wrong-password", stored["password_hash"]))
        self.assertEqual(auth.authenticate("ALICE@example.test", "correct-horse-battery")["id"], self.alice["id"])

        token = auth.create_session(self.alice["id"])
        self.assertEqual(auth.get_session_user(token)["id"], self.alice["id"])
        auth.delete_session(token)
        self.assertIsNone(auth.get_session_user(token))

    def test_upload_and_run_history_are_owner_scoped(self):
        database.add_upload("/tmp/alice.jpg", "healthy", self.alice["id"])
        database.add_upload("/tmp/bob.jpg", "bean_rust", self.bob["id"])

        self.assertEqual(database.count_pending(self.alice["id"]), 1)
        self.assertEqual(database.count_pending(self.bob["id"]), 1)
        self.assertEqual(
            [row["path"] for row in database.get_pending_uploads(self.alice["id"])],
            ["/tmp/alice.jpg"],
        )
        self.assertEqual(
            [row["path"] for row in database.get_all_uploads(self.bob["id"])],
            ["/tmp/bob.jpg"],
        )
        self.assertEqual(database.count_all_uploads(self.alice["id"]), 1)

        database.add_run(
            self.alice["id"], database.now(), 1, 0.8, 0.9, True, "promoted",
            {"accuracy": 0.8}, {"accuracy": 0.9, "f1": 0.89},
        )
        self.assertEqual(len(database.recent_runs(self.alice["id"])), 1)
        self.assertEqual(database.recent_runs(self.bob["id"]), [])
        self.assertEqual(database.recent_runs(self.alice["id"])[0]["new_metrics"]["f1"], 0.89)

    def test_only_one_replica_can_claim_shared_retraining(self):
        owners = [self.alice["id"], self.bob["id"]]
        with ThreadPoolExecutor(max_workers=2) as executor:
            job_ids = list(executor.map(database.claim_retraining, owners))
        self.assertEqual(sum(job_id is not None for job_id in job_ids), 1)
        winner_index = next(index for index, job_id in enumerate(job_ids) if job_id is not None)
        owner_id = owners[winner_index]
        job_id = job_ids[winner_index]
        assert job_id is not None

        database.finish_retraining(owner_id, "stale-job-id", "done")
        self.assertTrue(database.retraining_active())
        database.finish_retraining(owner_id, job_id, "done")
        self.assertFalse(database.retraining_active())

    def test_expired_retraining_job_is_cleared(self):
        job_id = database.claim_retraining(self.alice["id"])
        self.assertIsNotNone(job_id)
        with database.connect() as db:
            db.execute(
                "UPDATE retrain_jobs SET claimed_at = ? WHERE singleton_id = 1",
                ((datetime.now().astimezone() - timedelta(minutes=3)).isoformat(),),
            )

        self.assertFalse(database.retraining_active())
        self.assertIsNotNone(database.claim_retraining(self.bob["id"]))

    def test_retraining_heartbeat_keeps_job_active(self):
        job_id = database.claim_retraining(self.alice["id"])
        self.assertIsNotNone(job_id)

        self.assertTrue(database.renew_retraining(self.alice["id"], job_id))
        self.assertTrue(database.retraining_active())
        self.assertFalse(database.renew_retraining(self.bob["id"], job_id))

    def test_existing_database_is_migrated(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        with sqlite3.connect(legacy_path) as db:
            db.execute("""CREATE TABLE uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, label TEXT NOT NULL,
                uploaded_at TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0)""")
            db.execute("""CREATE TABLE retrain_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, finished_at TEXT,
                new_images INTEGER, old_accuracy REAL, new_accuracy REAL,
                promoted INTEGER, message TEXT)""")
        with patch.object(database, "DB_PATH", legacy_path):
            database.init_db()
            with database.connect() as db:
                upload_columns = {row["name"] for row in db.execute("PRAGMA table_info(uploads)")}
                run_columns = {row["name"] for row in db.execute("PRAGMA table_info(retrain_runs)")}
                job_tables = {
                    row["name"]
                    for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                }
        self.assertIn("owner_id", upload_columns)
        self.assertIn("owner_id", run_columns)
        self.assertIn("new_metrics", run_columns)
        self.assertIn("retrain_jobs", job_tables)


if __name__ == "__main__":
    unittest.main()
