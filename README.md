# Sales Forecasting – 90-day daily forecast

End-to-end ML project: raw order lines → daily sales → 36 regression models (35 ran; RANSAC failed) compared with time-based back-testing →
FastAPI (Docker, Render) → Streamlit dashboard.

```
sales_forecasting_project/
├── data/                       New_Query_2026_10_22_10_01_37.csv  (raw order lines)
├── notebook/
│   ├── Sales_Forecasting.ipynb the full analysis (already executed – outputs are saved)
│   └── nb_part*.py, build_and_run.py   (source of the notebook; only needed to rebuild it)
├── api/                        BACKEND (Docker / Render)
│   ├── main.py                 FastAPI app
│   ├── forecast_core.py        feature engineering shared by notebook + API (no training/serving skew)
│   ├── artifacts/              model.joblib, metadata.json (version, metrics), history.csv, forecast.csv
│   ├── requirements.txt  Dockerfile  .dockerignore  .env.example
├── frontend/                   FRONTEND (Streamlit)
│   ├── app.py  requirements.txt  Dockerfile  .streamlit/config.toml
├── docs/                       documentation, figures (PNG), result tables (CSV/JSON)
├── render.yaml                 Render blueprint for both services
└── README.md
```

## 1. Re-run the notebook (optional – it is already executed)
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib jupyter
cd notebook && jupyter notebook Sales_Forecasting.ipynb     # Run All  (≈ 4–5 min)
```
It rewrites `api/artifacts/*` and `docs/*`. Python 3.12 / scikit-learn 1.8 were used – keep the same scikit-learn
version in `api/requirements.txt` as the one that trained the model (a joblib file is version-sensitive).

## 2. Run the backend locally
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000        # docs: http://localhost:8000/docs
```
| Endpoint | Purpose |
|---|---|
| `GET /health` | health check (200 = model loaded, 503 = problem) |
| `GET /model-info` | version, features, metrics, error bands, feature importance |
| `GET /forecast?days=90` | daily forecast + 5–95 % band + summary (days 1–180) |
| `GET /forecast/date?date=2014-10-15` | one day (past dates return the actual sales) |
| `GET /forecast/range?start=…&end=…` | totals for a custom period |
| `GET /history?days=365` | historical daily sales |
| `GET /leaderboard` | top model combinations from the comparison |

Production features: input validation (422 with a clear message), JSON error handling, request-ID logging, `/health`,
environment variables (`MODEL_DIR`, `MAX_HORIZON`, `ALLOWED_ORIGINS`, `LOG_LEVEL` – see `.env.example`), model versioning
(`model_version`, `trained_at`, `data_hash` in `metadata.json`, returned by `/model-info`), pinned `requirements.txt`, non-root Dockerfile with `HEALTHCHECK`.

## 3. Run the frontend locally
```bash
cd frontend
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py          # Windows PowerShell:  $env:API_URL="http://localhost:8000"
```

## 4. Docker
```bash
docker build -t sales-api ./api   && docker run -p 8000:8000 sales-api
docker build -t sales-ui  ./frontend && docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 sales-ui
```

## 5. Deploy on Render
1. Push this folder to a GitHub repository.
2. Render → **New → Blueprint** → select the repo (uses `render.yaml`), or create two *Web Services* manually with **Runtime = Docker**:
   * API: Dockerfile path `./api/Dockerfile`, Docker context `./api`, health-check path `/health`.
   * UI: Dockerfile path `./frontend/Dockerfile`, Docker context `./frontend`, env var `API_URL` = the public URL of the API service.
3. After the API is live, copy its URL (e.g. `https://sales-forecast-api.onrender.com`) into the UI service's `API_URL` and redeploy the UI.
4. Free-tier services sleep when idle – the first request can take ~30–60 s (the UI retries automatically).

> The assignment text mentions AWS; this project follows the requested **Render + Docker** setup. The same Docker images run unchanged on AWS (ECS / App Runner).

## 6. Updating the forecast with new data
Replace the CSV in `data/`, re-run the notebook (it re-selects the best model), commit the new `api/artifacts/` and redeploy. `metadata.json` gets a new `data_hash` and `trained_at`; bump `model_version` in the notebook's save cell.

## 7. How the forecast works (one paragraph)
Order lines are summed to one value per calendar day (missing days = 0). The model uses date-only features (trend counter, month/week/weekday,
sine/cosine seasonality, month-end flags…), so it can forecast any future day directly. The selected regression model is a Lasso regression.
Uncertainty bands come from the real out-of-sample errors of four 90-day back-test windows.
