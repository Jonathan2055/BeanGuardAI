import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config, model_versions, prediction


class ModelVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.models_dir = self.root / "models"
        self.models_dir.mkdir()
        self.version_dir = self.models_dir / "versions"
        self.current_model = self.models_dir / "bean_model.keras"
        self.current_h5 = self.models_dir / "bean_model.h5"
        self.current_metrics = self.models_dir / "metrics.json"
        self.current_model.write_bytes(b"initial-model")
        self.current_h5.write_bytes(b"initial-inference-model")
        self.current_metrics.write_text(json.dumps({"accuracy": 0.91}))
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

    def tearDown(self):
        self.version_dir_patch.stop()
        self.config_patch.stop()
        self.temp_dir.cleanup()

    def test_initial_version_is_a_write_once_snapshot(self):
        model_versions.ensure_initial_version()
        self.current_model.write_bytes(b"changed-model")
        self.current_h5.write_bytes(b"changed-inference-model")
        self.current_metrics.write_text(json.dumps({"accuracy": 0.99}))

        model_versions.ensure_initial_version()

        self.assertEqual(model_versions.model_path("v1").read_bytes(), b"initial-model")
        self.assertEqual(
            (self.version_dir / "v1.h5").read_bytes(),
            b"initial-inference-model",
        )
        self.assertEqual(model_versions.read_metrics("v1"), {
            "accuracy": 0.91,
            "version": 1,
            "evaluation_split": "test",
        })

    def test_new_versions_are_immutable_and_listed_in_order(self):
        model_versions.ensure_initial_version()
        candidate_model = self.root / "candidate.keras"
        candidate_inference = self.root / "candidate.h5"
        candidate_model.write_bytes(b"candidate-model")
        candidate_inference.write_bytes(b"candidate-inference-model")

        version = model_versions.save_version(
            candidate_model,
            candidate_inference,
            {"accuracy": 0.95, "evaluation_split": "validation"},
        )

        self.assertEqual(version, "v2")
        self.assertEqual(model_versions.model_path("v2").read_bytes(), b"candidate-model")
        self.assertEqual(model_versions.read_metrics("v2")["version"], 2)
        versions = model_versions.available_versions("v2")
        self.assertEqual([item["id"] for item in versions], ["v1", "v2"])
        self.assertEqual([item["active"] for item in versions], [False, True])
        with self.assertRaises(FileNotFoundError):
            model_versions.model_path("v3")
        with self.assertRaises(ValueError):
            model_versions.model_path("current")

    def test_prediction_loads_the_requested_version(self):
        model_versions.ensure_initial_version()
        candidate_model = self.root / "candidate.keras"
        candidate_inference = self.root / "candidate.h5"
        candidate_model.write_bytes(b"candidate-model")
        candidate_inference.write_bytes(b"candidate-inference-model")
        model_versions.save_version(
            candidate_model,
            candidate_inference,
            {"accuracy": 0.95},
        )
        with patch.object(prediction, "_models", {}), patch.object(
            prediction.keras.models,
            "load_model",
            side_effect=["initial-loaded", "candidate-loaded"],
        ) as load_model:
            self.assertEqual(prediction.get_model("v1"), "initial-loaded")
            self.assertEqual(prediction.get_model("v2"), "candidate-loaded")

        self.assertEqual(
            [call.args[0] for call in load_model.call_args_list],
            [self.version_dir / "v1.keras", self.version_dir / "v2.keras"],
        )


if __name__ == "__main__":
    unittest.main()
