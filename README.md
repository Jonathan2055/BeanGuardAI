# BeanGuard AI

Screen a photo of a bean leaf for **angular leaf spot**, **bean rust**, or a **healthy** leaf. A MobileNetV2 model (transfer learning) is served by a FastAPI app with a web dashboard, where logged-in users can upload labelled photos and retrain the model.

Live app : <http://51.20.10.95/> . [API docs (Swagger)](http://51.20.10.95/docs) 
Demo video : TODO: YouTube link 
Dataset : [Bean Disease Dataset on Kaggle](https://www.kaggle.com/datasets/therealoise/bean-disease-dataset) 
Notebook : [notebook/bean_guard_notebook.ipynb](notebook/bean_guard_notebook.ipynb) 

## Problem

Angular leaf spot and bean rust are key bean diseases in Uganda and the wider region. They reduce yields, and farmers often have to wait for an expert to identify them. BeanGuard gives a first screening from a phone photo. It is a **screening aid, not a diagnosis**: confirm with an extension officer before treating a field.

## Features

- **Predict:** upload one leaf photo and get the class, the confidence and all class probabilities. Use a close photo of a single bean leaf (see [Limitations](#limitations)).
- **Dashboard:** API up-time, model version, test-set metrics, confusion matrix and retraining history.
- **Model versions:** each successful retrain keeps the former current model as a previous version; logged-in users can choose which saved version serves predictions. Retraining always starts from the current model.
- **Data insights:** class balance, average colour per class and per-class F1.
- **Upload and retrain** (login required): upload several images of one class, or a `.zip` with one folder per class, then start retraining. Retraining runs in the background.

## Results

Evaluated once on the held-out test split (153 images, 51 per class) after all choices were made on the validation split.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline CNN (from scratch) | 71.9% | 72.3% | 71.9% | 71.5% |
| **MobileNetV2, fine-tuned** | **89.5%** | **89.8%** | **89.5%** | **89.6%** |

Precision, recall and F1 are macro averages. Per-class F1: angular leaf spot 0.89, bean rust 0.85, healthy 0.95. The final model makes 16 errors on 153 images. Of the 102 diseased leaves, 100 were flagged as diseased, but only 89 got the exact disease right. Bean rust and angular leaf spot are mostly confused with each other.

With only 153 test images, treat these scores as roughly ±5 points. Full training, optimisation steps and interpretation are in the notebook.

## Repository layout

```
├── notebook/bean_guard_notebook.ipynb   data preparation, training, evaluation
├── src/
│   ├── preprocessing.py   image loading, validation, augmentation
│   ├── model.py           model, fine-tuning, evaluation helpers
│   ├── prediction.py      load the saved model and predict one image
│   ├── retrain.py         retraining pipeline
│   ├── auth.py            accounts and sessions
│   ├── database.py        SQLite: users, uploads, retraining runs
│   ├── api.py             FastAPI app
│   └── config.py          paths and settings
├── ui/                    index.html, app.js, styles.css
├── data/                  train/ val/ test/ (from the notebook), uploads/, app.db
├── models/                bean_model.keras, bean_model.h5, metrics.json
├── tests/                 authentication and database tests
├── Dockerfile, docker-compose.yml, nginx.conf, Caddyfile
├── locustfile.py          load test
└── requirements.txt
```

## Setup

**Requirements:** Python **3.13**. TensorFlow 2.21 supports Python 3.10 to 3.13, so **do not use Python 3.14**: `pip install tensorflow` fails there.

```bash
git clone https://github.com/Jonathan2055/BeanGuardAI.git && cd BeanGuardAI
python3.13 --version                      # must print 3.13.x because the tensorflow doesn't have a released updates that works with python 3.14
python3.13 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export APP_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> (API docs at `/docs`). Register an account to upload data or retrain. `APP_SECRET_KEY` must be at least 32 random characters. Run the tests with `python -m pytest tests`.

The repository already contains a trained model in `models/`, so you only need the notebook to retrain from scratch.

## Train the model (notebook)

Open `notebook/bean_guard_notebook.ipynb` on Kaggle, attach the *Bean Disease Dataset*, enable a GPU and run all cells. It cleans the images, makes a stratified 70/15/15 split, trains the baseline and the MobileNetV2 model, and writes `models/bean_model.keras` (used by the API and retraining), `models/bean_model.h5` (inference-only copy), `models/metrics.json` and `data/class_names.json`. It also runs locally if the dataset is in `data/raw_data/`.

## Docker

```bash
cat .env          # set APP_SECRET_KEY: (with 32+ characters); keep COOKIE_SECURE=false for local HTTP
docker compose up -d --build
curl http://localhost:8080/health
```

Open <http://localhost:8080>. `data/` and `models/` are bind-mounted, so the database, uploads and models survive rebuilds. Back them up separately. Stop with `docker compose down` (data is kept).

**HTTPS:** set `APP_DOMAIN` to a real hostname and `COOKIE_SECURE=true` in `.env`, open TCP ports 80 and 443, then run `docker compose --profile https up -d --build`. Caddy obtains and renews the certificate.

## Deployment (AWS)

Deployed on an AWS EC2 instance (TODO: region and instance type) with Docker Compose and HTTPS.

1. Launch an Ubuntu instance with at least 4 GB RAM. Allow inbound TCP 22 (your IP only), 80 and 443.
2. Install Docker Engine and the Compose plugin, then clone this repository.
3. Make sure `models/` and `data/` (train, val, test, `class_names.json`) are present on the instance.
4. Create `.env` as above, then run `docker compose --profile https up -d --build`.
5. Check `https://<your-domain>/health`.

## Load test (Locust)

`locustfile.py` posts held-out leaf images to `/predict`. Run it from the repository root against the same host and with the same settings for every run, changing only the number of API containers:

```bash
mkdir -p results
docker compose up -d --scale api=1        # then 2 and 3
locust -f locustfile.py --headless -u 20 -r 4 -t 2m \
  --host http://localhost:8080 --csv results/containers_1 --html results/containers_1.html
```

| API containers | Users | Requests | Failures | Median (ms) | p95 (ms) | Requests/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | TODO | TODO | TODO | TODO | TODO |
| 2 | 20 | TODO | TODO | TODO | TODO | TODO |
| 3 | 20 | TODO | TODO | TODO | TODO | TODO |

TODO: two or three sentences on how latency and throughput changed. Containers on one machine share its CPU, so more containers do not always mean proportionally faster responses.

## Limitations

- Trained on 990 field photos from one dataset. Photos from other phones, lighting or backgrounds may score lower than the test split.
- **Photos that are not bean leaves are still classified.** The app only checks that a file is a valid image. If someone uploads a photo of something else (a person, a different plant, soil), the model still answers with one of its three classes, with a confidence that can look high. The model has only ever seen bean leaves, so it has no idea of other images. We tried to add an input check that compares a photo's internal model features with the training leaves, but it did not help in our tests (non-bean photos were still accepted), so we did not continue with it and removed it. Only upload close photos of a single bean leaf.
- Retraining trusts the labels users upload. Wrong labels can make the model worse.
- SQLite and local files suit a single server, not large-scale use.

## Credits

Images: Bean Disease Dataset on Kaggle, which matches the Makerere AI Lab "Beans" data (field photos from Uganda, annotated with NaCRRI). 
