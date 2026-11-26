import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from src import auth, database, model_versions


class ModelVersionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.database_path = Path(cls.temp_dir.name) / "app.db"
        cls.database_patch = patch.object(database, "DB_PATH", cls.database_path)
        cls.database_patch.start()
        cls.secret_patch = patch.dict(os.environ, {"APP_SECRET_KEY": "test-key-" + "x" * 40})
        cls.secret_patch.start()
        database.init_db()

        from src import api

        cls.api = api

    @classmethod
    def tearDownClass(cls):
        cls.secret_patch.stop()
        cls.database_patch.stop()
        cls.temp_dir.cleanup()

    def setUp(self):
        self.models_dir = Path(tempfile.mkdtemp(dir=self.temp_dir.name))
        self.current_model = self.models_dir / "bean_model.keras"
        self.previous_model = self.models_dir / "bean_model_previous.keras"
        self.active_model = self.models_dir / "active_model.json"
        self.current_model.write_bytes(b"current")
        self.previous_model.write_bytes(b"previous")
        self.model_paths_patch = patch.multiple(
            model_versions,
            MODEL_PATHS={
                model_versions.CURRENT: self.current_model,
                model_versions.PREVIOUS: self.previous_model,
            },
            ACTIVE_MODEL_PATH=self.active_model,
        )
        self.model_paths_patch.start()
        self.user = auth.create_account(
            f"{self._testMethodName}@example.test",
            "Switch",
            "a-long-test-password",
        )

    def tearDown(self):
        self.model_paths_patch.stop()
        for path in self.models_dir.iterdir():
            path.unlink()
        self.models_dir.rmdir()

    def request(self, token=None):
        headers = []
        if token:
            headers.append((b"cookie", f"{auth.SESSION_COOKIE}={token}".encode()))
        return Request({
            "type": "http",
            "method": "POST",
            "path": "/model-version",
            "headers": headers,
        })

    def test_selecting_model_version_requires_login_and_persists_selection(self):
        with self.assertRaises(HTTPException) as unauthenticated:
            self.api.select_model_version(
                self.api.ModelVersionPayload(version="previous"),
                self.request(),
            )
        self.assertEqual(unauthenticated.exception.status_code, 401)

        token = auth.create_session(self.user["id"])
        response = self.api.select_model_version(
            self.api.ModelVersionPayload(version="previous"),
            self.request(token),
        )
        self.assertEqual(response, {"active_model": "previous"})
        self.assertEqual(model_versions.get_active_version(), model_versions.PREVIOUS)

    def test_selecting_unknown_or_missing_model_returns_client_error(self):
        token = auth.create_session(self.user["id"])
        with self.assertRaises(HTTPException) as unknown:
            self.api.select_model_version(
                self.api.ModelVersionPayload(version="other"),
                self.request(token),
            )
        self.assertEqual(unknown.exception.status_code, 400)

        self.previous_model.unlink()
        with self.assertRaises(HTTPException) as missing:
            self.api.select_model_version(
                self.api.ModelVersionPayload(version="previous"),
                self.request(token),
            )
        self.assertEqual(missing.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
