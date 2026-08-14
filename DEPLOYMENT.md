# Deployment Guide

This project ships as a single Docker image, so it deploys the same way to
Render, Railway, Fly.io, or any container host. Below are step-by-step
instructions for **Render** and **Railway**, the two most common free/cheap
options for portfolio projects.

Before deploying, push this project to a **GitHub repository** — both
platforms deploy from a connected repo.

```bash
git init
git add .
git commit -m "Heart disease risk prediction API"
git branch -M main
git remote add origin https://github.com/<your-username>/heart-disease-risk-api.git
git push -u origin main
```

> Make sure `artifacts/model.pkl` and `artifacts/OneHotEncoder.pkl` are
> committed — the API cannot start without them. If your repo has a
> `.gitignore` that excludes `*.pkl`, remove that rule for this project.

---

## Option A: Render

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New +** → **Blueprint**, and select this repository. Render
   will detect `render.yaml` and pre-fill the service configuration.
   - Alternatively: **New +** → **Web Service** → select the repo →
     set **Environment** to `Docker` (Render will find the `Dockerfile`
     automatically).
3. Confirm settings:
   - **Environment**: Docker
   - **Health Check Path**: `/health`
   - **Instance Type**: Free (fine for a demo/portfolio project)
4. Add environment variables (from `.env.example`) under **Environment**
   if you didn't use the Blueprint, especially:
   - `MODEL_VERSION`
   - `ENABLE_MLFLOW_LOGGING` (set to `false` if you don't want an MLflow
     backend running here — see note below)
5. Click **Create Web Service**. Render builds the Docker image and
   deploys it; the first build takes a few minutes (TensorFlow is a large
   dependency).
6. Once live, your API is available at
   `https://<your-service-name>.onrender.com`. Test it:
   ```bash
   curl https://<your-service-name>.onrender.com/health
   ```
   and open `https://<your-service-name>.onrender.com/dashboard`.

**Notes for Render free tier:**
- Free instances spin down after inactivity; the first request after
  idling will be slow (cold start).
- The free tier's filesystem is **ephemeral** — the SQLite monitoring DB
  and local `mlruns/` folder reset on every deploy/restart. For a
  persistent dashboard history, either upgrade to a paid instance with a
  persistent disk, or point `MLFLOW_TRACKING_URI` at a hosted MLflow
  server (see below).

---

## Option B: Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo** → select this
   repository.
3. Railway auto-detects the `Dockerfile` and builds from it. If it
   instead tries to use Nixpacks, explicitly set **Settings → Build →
   Builder** to `Dockerfile`.
4. Under **Variables**, add the same environment variables as in
   `.env.example`. Railway automatically provides `PORT` — the app
   already reads it (`app.main` binds to `0.0.0.0:$PORT` via the
   `Dockerfile`/`Procfile`).
5. Under **Settings → Networking**, click **Generate Domain** to get a
   public URL.
6. Deploy. Once live:
   ```bash
   curl https://<your-app>.up.railway.app/health
   ```

**Notes for Railway:**
- Railway's default filesystem is also ephemeral between deploys. Add a
  **Volume** (Settings → Volumes) mounted at `/srv/data` if you want the
  SQLite monitoring log to persist across deploys.

---

## Persisting MLflow tracking data (optional, either platform)

For a demo, `MLFLOW_TRACKING_URI=file:./mlruns` (the default) is fine —
it's disabled the moment the container restarts on a free tier. For a
persistent, always-available MLflow UI:

1. Deploy a second service running the official MLflow server image
   (`ghcr.io/mlflow/mlflow`) with a persistent volume or an external
   database (e.g. a free Postgres instance) as the backend store.
2. Point the API's `MLFLOW_TRACKING_URI` environment variable at that
   service's URL, e.g. `https://mlflow-service.onrender.com`.
3. Re-deploy the API — no code changes needed, this is entirely
   configuration-driven (see `app/config.py`).

`docker-compose.yml` in this repo shows the same pattern for local
development, if you want to test it before deploying.

---

## Verifying a deployment

After deploying anywhere, run through this checklist:

```bash
BASE_URL=https://<your-deployed-url>

curl $BASE_URL/health
curl -X POST $BASE_URL/predict -H "Content-Type: application/json" -d @sample_request.json
curl $BASE_URL/stats
```

(See the `README.md` "API usage" section for a full sample request body,
or use `sample_request.json` if you saved one.) Then open
`$BASE_URL/dashboard` in a browser to confirm the request you just sent
shows up in the monitoring table.
