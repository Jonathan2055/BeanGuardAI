import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import model_versions, prediction


class ModelVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.current_model = self.root / "bean_model.keras"
        self.previous_model = self.root / "bean_model_previous.keras"
        self.current_h5 = self.root / "bean_model.h5"
        self.previous_h5 = self.root / "bean_model_previous.h5"
        self.current_metrics = self.root / "metrics.json"
        self.previous_metrics = self.root / "metrics_previous.json"
        self.active_model = self.root / "active_model.json"
        self.paths_patch = patch.multiple(
            model_versions,
            MODEL_PATHS={
                model_versions.CURRENT: self.current_model,
                model_versions.PREVIOUS: self.previous_model,
            },
            INFERENCE_MODEL_PATHS={
                model_versions.CURRENT: self.current_h5,
                model_versions.PREVIOUS: self.previous_h5,
            },
            METRICS_PATHS={
                model_versions.CURRENT: self.current_metrics,
                model_versions.PREVIOUS: self.previous_metrics,
            },
            ACTIVE_MODEL_PATH=self.active_model,
        )
        self.paths_patch.start()
        self.models_dir_patch = patch.object(model_versions.config, "MODELS_DIR", self.root)
        self.models_dir_patch.start()

    def tearDown(self):
        self.models_dir_patch.stop()
        self.paths_patch.stop()
        self.temp_dir.cleanup()

    def test_preserve_current_snapshots_models_and_metrics(self):
        self.current_model.write_bytes(b"current-model")
        self.current_h5.write_bytes(b"current-inference-model")
        current_metrics = {"version": 4, "accuracy": 0.91}
        self.current_metrics.write_text(json.dumps(current_metrics))

        model_versions.preserve_current_version()

        self.assertEqual(self.previous_model.read_bytes(), b"current-model")
        self.assertEqual(self.previous_h5.read_bytes(), b"current-inference-model")
        self.assertEqual(model_versions.read_metrics(model_versions.PREVIOUS), current_metrics)

    def test_active_version_defaults_to_current_and_selection_requires_saved_model(self):
        self.current_model.write_bytes(b"model")
        self.assertEqual(model_versions.get_active_version(), model_versions.CURRENT)

        with self.assertRaises(FileNotFoundError):
            model_versions.set_active_version(model_versions.PREVIOUS)

        self.previous_model.write_bytes(b"previous")
        model_versions.set_active_version(model_versions.PREVIOUS)
        self.assertEqual(model_versions.get_active_version(), model_versions.PREVIOUS)
        versions = model_versions.available_versions()
        self.assertEqual([version["id"] for version in versions], [
            model_versions.CURRENT,
            model_versions.PREVIOUS,
        ])
        self.assertEqual([version["active"] for version in versions], [False, True])

    def test_prediction_loads_the_selected_model_file(self):
        self.current_model.write_bytes(b"current")
        self.previous_model.write_bytes(b"previous")
        with patch.object(prediction, "_models", {}), patch.object(
            prediction.keras.models,
            "load_model",
            side_effect=["current-loaded", "previous-loaded"],
        ) as load_model:
            model_versions.set_active_version(model_versions.CURRENT)
            self.assertEqual(prediction.get_model(), "current-loaded")

            model_versions.set_active_version(model_versions.PREVIOUS)
            self.assertEqual(prediction.get_model(), "previous-loaded")

        self.assertEqual(
            [call.args[0] for call in load_model.call_args_list],
            [self.current_model, self.previous_model],
        )
