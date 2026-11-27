# BeanGuard AI

BeanGuard screens bean-leaf photos for **angular leaf spot**, **bean rust**, or **healthy** using a fine-tuned MobileNetV2 model. It is an educational screening aid, not a diagnosis.

- **Live app:** [51.20.10.95](http://51.20.10.95/) · 
- **[API (Swagger)](http://51.20.10.95/docs)** · 
- **[Demo video](https://youtu.be/sbu2-QBdUBs)** · 
- **[Training notebook](notebook/bean_guard_notebook.ipynb)** · 
- **[Dataset](https://www.kaggle.com/datasets/therealoise/bean-disease-dataset)**- 

## Features

- Predict a class, confidence, and per-class probabilities from a bean-leaf photo.
- Explore model metrics, class balance, colour insights, and retraining history.
- Register to upload labeled images, retrain, and select your model version.
- Inspect or call the API through Swagger at `/docs`.

## Install and run with Docker

1. Install Git, Docker Engine, and the Docker Compose plugin.
2. Clone the repository and enter it:
   ```bash
   git clone https://github.com/Jonathan2055/BeanGuardAI.git
   cd BeanGuardAI
   ```
3. Create local settings with a unique secret:
   ```bash
   printf 'APP_SECRET_KEY=%s\nCOOKIE_SECURE=false\nAPP_DOMAIN=localhost\n' \
     "$(openssl rand -hex 32)" > .env
   ```
4. Build and start the app:
   ```bash
   docker compose up -d --build
   docker compose ps
   curl http://localhost:8080/health
   ```
5. Open [the dashboard](http://localhost:8080/) or [Swagger](http://localhost:8080/docs). Stop with `docker compose down`; data and models remain on disk.

The repository includes the trained model and dataset splits. Docker mounts `data/` and `models/` for persistence. The default local setup uses HTTP; for a real domain, set `APP_DOMAIN` and `COOKIE_SECURE=true`, open ports 80/443, and start with `docker compose --profile https up -d --build`.

## Dashboard guide

1. **Overview:** check service/model status, evaluation metrics, and your retraining history.
2. **Image prediction:** upload a clear, close-up JPEG or PNG of one bean leaf (up to 10 MB), then select **Analyze image**. The result shows the predicted class and probabilities.
3. **Data insights:** review training class counts, average image colours, and per-class performance.
4. **Upload & retrain:** register or log in, then upload images individually under their correct class or as a ZIP with one folder per class. Review the pending count and start retraining (1–20 epochs). Results appear in the dashboard. Incorrect labels may reduce model quality.

Only bean-leaf photos are suitable: other images are still assigned one of the model's three classes. Verify important decisions with an agronomist.

## API quick guide

Open `/docs` and use **Try it out**. Register or log in first for account-only operations; the session cookie is set automatically.

| Endpoint | What to provide / purpose |
|---|---|
| `POST /predict` | Multipart field `file`: one JPEG/PNG bean-leaf image, max 10 MB. Returns the prediction and probabilities. Works without login. |
| `POST /upload` | Login required. Multipart `label` (one of the listed classes) and `files` (1–100 JPEG/PNG images, max 10 MB each); queues labeled data for your retraining. |
| `POST /upload-zip` | Login required. Multipart `file`: ZIP with class-named folders at its root, e.g. `healthy/leaf.jpg`. Max 500 MB compressed. |
| `POST /retrain` | Login required and pending uploads required. JSON `{"epochs": 5}` starts a background run (1–20 epochs; default 5). |
| `POST /model-version` | Login required. JSON `{"version": "v1"}` selects an available model for your account. |
| `GET /status`, `GET /insights` | Read model/service status and dataset summaries. |

## Results

Held-out test split (153 images; 51 per class):

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline CNN | 71.9% | 71.5% |
| Fine-tuned MobileNetV2 | **89.5%** | **89.6%** |

Scores are based on a small test set and may not generalize to other phones, lighting, or backgrounds. See the notebook for data preparation, training, and detailed evaluation.

## Local development and load testing

For local development, use Python 3.13 (TensorFlow does not support Python 3.14 in this setup):

```bash
python3.13 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export APP_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Run tests with `python -m pytest tests`. For load testing, start the Docker app above, then install and run Locust from the repository root. Using the virtual-environment Python avoids relying on a globally installed `locust` command:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install locust
mkdir -p results
.venv/bin/python -m locust -f locustfile.py --headless -u 20 -r 4 -t 2m \
  --host http://localhost:8080 --csv results/containers_1 --html results/containers_1.html
```

The workload uses a held-out image from `data/test/healthy`. The API is intended to run as one container with SQLite/local-file storage. Existing multi-container measurements were inconsistent and do not support a reliable scaling comparison.

To retrain from scratch, open `notebook/bean_guard_notebook.ipynb`, attach the Bean Disease Dataset, and run all cells.
