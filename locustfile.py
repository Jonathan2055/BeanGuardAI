"""Model inference workload for Locust.

Run from the repository root after starting the API:
  locust -f locustfile.py --headless -u 20 -r 4 -t 2m --host http://localhost
"""
import os
from pathlib import Path

from locust import HttpUser, between, task


def load_sample():
    configured = os.getenv("LOCUST_IMAGE")
    if configured:
        image_path = Path(configured)
    else:
        image_dir = Path("data/test/healthy")
        image_path = next(
            (path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            None,
        )
    if image_path is None or not image_path.is_file():
        raise FileNotFoundError("Set LOCUST_IMAGE or provide data/test/healthy sample images.")
    return image_path.name, image_path.read_bytes()


class PredictionUser(HttpUser):
    wait_time = between(0.05, 0.2)
    image_name, image_bytes = load_sample()

    @task
    def predict_leaf(self):
        content_type = "image/png" if self.image_name.lower().endswith(".png") else "image/jpeg"
        with self.client.post(
            "/predict",
            files={"file": (self.image_name, self.image_bytes, content_type)},
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return
            try:
                result = response.json()
            except ValueError:
                response.failure("Response was not valid JSON")
                return
            if "class" not in result or "probabilities" not in result:
                response.failure("Prediction response is missing expected fields")
