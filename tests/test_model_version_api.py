import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from src import auth, config, database, model_versions


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
        self.root = Path(self.temp_dir.name) / self._testMethodName
        self.models_dir = self.root / "models"
        self.models_dir.mkdir(parents=True)
        self.version_dir = self.models_dir / "versions"
        self.current_model = self.models_dir / "bean_model.keras"
        self.current_h5 = self.models_dir / "bean_model.h5"
        self.current_model.write_bytes(b"initial-model")
        self.current_h5.write_bytes(b"initial-inference-model")
        (self.models_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.8}))
        self.config_patch = patch.multiple(
            config,
            MODELS_DIR=self.models_dir,
            MODEL_PATH=self.current_model,
            MODEL_H5_PATH=self.current_h5,
        )
        self.config_patch.start()
        self.version_dir_patch = patch.object(
            model_versions, "VERSION_DIR", self.version_dir
        )
        self.version_dir_patch.start()
        model_versions.ensure_initial_version()
        candidate_model = self.root / "candidate.keras"
        candidate_inference = self.root / "candidate.h5"
        candidate_model.write_bytes(b"candidate-model")
        candidate_inference.write_bytes(b"candidate-inference-model")
        model_versions.save_version(
            candidate_model,
            candidate_inference,
            {"accuracy": 0.9},
        )
        self.user = auth.create_account(
            f"{self._testMethodName}@example.test",
            "Switch",
            "a-long-test-password",
        )
        self.other_user = auth.create_account(
            f"other-{self._testMethodName}@example.test",
            "Other",
            "another-long-password",
        )

    def tearDown(self):
        self.version_dir_patch.stop()
        self.config_patch.stop()

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

    def test_selecting_model_version_requires_login_and_is_account_scoped(self):
        with self.assertRaises(HTTPException) as unauthenticated:
            self.api.select_model_version(
                self.api.ModelVersionPayload(version="v2"),
                self.request(),
            )
        self.assertEqual(unauthenticated.exception.status_code, 401)

        token = auth.create_session(self.user["id"])
        response = self.api.select_model_version(
            self.api.ModelVersionPayload(version="v2"),
            self.request(token),
        )
        self.assertEqual(response, {"active_model": "v2"})
        self.assertEqual(
            database.user_by_id(self.user["id"])["active_model_version"],
            "v2",
        )
        self.assertEqual(
            database.user_by_id(self.other_user["id"])["active_model_version"],
            "v1",
        )

    def test_selecting_unknown_or_missing_model_returns_client_error(self):
        token = auth.create_session(self.user["id"])
        with self.assertRaises(HTTPException) as unknown:
            self.api.select_model_version(
                self.api.ModelVersionPayload(version="current"),
                self.request(token),
            )
        self.assertEqual(unknown.exception.status_code, 400)

        (self.version_dir / "v2.keras").unlink()
        with self.assertRaises(HTTPException) as missing:
            self.api.select_model_version(
                self.api.ModelVersionPayload(version="v2"),
                self.request(token),
            )
        self.assertEqual(missing.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
