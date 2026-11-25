# BeanGuard AI

Bean-leaf disease classifier for angular leaf spot, bean rust and healthy leaves. Includes a Jupyter training notebook, prediction API, and HTML/CSS/JavaScript dashboard.

**Demo video (camera on):** Add link
**Live application:** Add HTTPS URL

## Results

On the 153-image test split: **89.54% accuracy**, **89.83% macro precision**, **89.54% macro recall**, **89.61% macro F1**. The baseline CNN reached 71.9% accuracy. Results may not generalize to other data.

## Run locally

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export APP_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. Register to upload correctly labeled bean-leaf photos or start manual retraining. The app checks image files but does not detect whether they contain bean leaves.

## Docker Compose

Install Docker Engine and the Compose plugin. Copy `.env.example` to `.env` and replace the placeholder `APP_SECRET_KEY` with a private random value of at least 32 characters. Keep `COOKIE_SECURE=false` for local HTTP.

```bash
cp .env.example .env
openssl rand -hex 32
```

Copy the generated value into `.env`, then build and start the containers:

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:8080/health
```

Open <http://localhost:8080>. `data/` and `models/` are bind-mounted so the SQLite database, uploads, and model files persist across container rebuilds. Back them up separately. To stop the containers, run `docker compose down`; this keeps the data.

If Docker reports permission denied for its socket on Linux, use Docker with the appropriate permissions (for example, `sudo docker compose ...` or the official Docker post-install steps). Do not make the Docker socket world-readable.

### Optional HTTPS

For a real DNS hostname, set `APP_DOMAIN` in `.env`, set `COOKIE_SECURE=true`, and allow inbound TCP 80/443 to the host. Then run `docker compose --profile https up -d --build`. Caddy obtains and renews the HTTPS certificate and stores its state in named Docker volumes. Do not enable this profile with the default `localhost` hostname for public deployment.

## Locust load test

`locustfile.py` posts a held-out leaf image to `/predict` and reports non-200 responses with their status and response body. Install Locust with `pip install locust` or use the dependency in `requirements.txt`.

Run against the local Compose app from the repository root:

```bash
mkdir -p results
locust -f locustfile.py --headless -u 20 -r 4 -t 2m \
  --host http://localhost:8080 --csv results/local --html results/local.html
```

Use the same image, user count, ramp-up, duration, and host machine when comparing runs. Only load-test services you own or have permission to test.

| Run | Requests shown | Failures | Median | p95 | Result |
|---|---:|---:|---:|---:|---|
| Submitted 20-user run (excerpt) | 5,558+ | 100% | 3 ms | Not provided | Failed; not a capacity result. The excerpt has no failure reason. |
| Local API smoke test (2 users, 10 s) | 28 | 0 | 540 ms | 790 ms | API reachable; not a Docker capacity benchmark. |

Rerun the failed test and save the complete Locust report before drawing performance conclusions.

## Notebook

Run [notebook/bean_guard_notebook.ipynb](notebook/bean_guard_notebook.ipynb) in Kaggle or Jupyter with the Bean Disease Dataset. It trains and evaluates the model and creates the artifacts in `models/`.
