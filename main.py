"""
Sales Forecasting API (FastAPI)
-------------------------------
Serves the 90-day daily sales forecast produced by the model trained in notebook/Sales_Forecasting.ipynb.

Environment variables (see .env.example):
    MODEL_DIR        folder with model.joblib / metadata.json / history.csv   (default: ./artifacts)
    MAX_HORIZON      longest forecast the API will produce, in days            (default: 180)
    ALLOWED_ORIGINS  comma-separated CORS origins, "*" = any                   (default: *)
    LOG_LEVEL        DEBUG | INFO | WARNING                                    (default: INFO)
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import forecast_core as fc

# ----------------------------------------------------------------------------- configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "artifacts"))
MAX_HORIZON = int(os.getenv("MAX_HORIZON", "180"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("sales-api")

STATE: dict = {}          # model, metadata, history and the pre-computed forecast live here


def _load_artifacts() -> None:
    """Load model + metadata + history and pre-compute the forecast once (a forecast of N days is the first N days of a longer one)."""
    t0 = time.time()
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    model = joblib.load(MODEL_DIR / "model.joblib")
    history = pd.read_csv(MODEL_DIR / "history.csv", index_col=0, parse_dates=True)["sales"].astype(float)
    capped = history.clip(upper=meta["cap_value"])
    pred = fc.forecast(model, capped, MAX_HORIZON, meta["use_lags"], meta["features"], pd.Timestamp(meta["origin"]),
                       max_value=meta["cap_value"])
    iv = meta["intervals"]["daily"]
    table = pd.DataFrame({"forecast": pred})
    table["lower"] = (table.forecast + iv["q05"]).clip(lower=0)
    table["upper"] = table.forecast + iv["q95"]
    STATE.update(meta=meta, model=model, history=history, table=table)
    log.info("Loaded model %s v%s (%s) in %.2fs; history %s → %s; forecast %s → %s",
             meta["model_name"], meta["model_version"], meta["feature_set"], time.time() - t0,
             history.index.min().date(), history.index.max().date(), table.index.min().date(), table.index.max().date())


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _load_artifacts()
    except Exception:                                  # the API still starts so /health can report the problem
        log.exception("Could not load model artifacts from %s", MODEL_DIR)
    yield


app = FastAPI(title="Sales Forecasting API", version="1.0.0", lifespan=lifespan,
              description="90-day daily sales forecast. Model trained with scikit-learn; see /docs for the interactive API.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET"], allow_headers=["*"])


# ----------------------------------------------------------------------------- middleware + error handling
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]; t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("[%s] unhandled error on %s", rid, request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Unexpected server error.", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    log.info("[%s] %s %s -> %s (%.0f ms)", rid, request.method, request.url.path, response.status_code, (time.time() - t0) * 1000)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    msgs = [f"{'.'.join(str(x) for x in e['loc'] if x != 'query')}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": "; ".join(msgs)})


@app.exception_handler(HTTPException)
async def http_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": "request_failed", "detail": exc.detail})


def _require_model() -> None:
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check the server logs and MODEL_DIR.")


def _clean(x: float) -> float:
    return 0.0 if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else round(float(x), 2)


def _rows(tbl: pd.DataFrame) -> list:
    return [{"date": d.strftime("%Y-%m-%d"), "weekday": d.day_name(), "forecast": _clean(r.forecast),
             "lower": _clean(r.lower), "upper": _clean(r.upper)} for d, r in tbl.iterrows()]


def _summary(tbl: pd.DataFrame) -> dict:
    n = len(tbl); total = float(tbl.forecast.sum()); iv = STATE["meta"]["intervals"]["90day"]
    scale = math.sqrt(n / 90.0)                                   # approximate: error of a sum grows ~ sqrt(days)
    peak = tbl.forecast.idxmax()
    return {"days": n, "total_forecast": _clean(total), "average_per_day": _clean(total / n),
            "total_lower": _clean(max(total + iv["q05"] * scale, 0)), "total_upper": _clean(total + iv["q95"] * scale),
            "highest_day": peak.strftime("%Y-%m-%d"), "highest_day_forecast": _clean(tbl.forecast.max()),
            "lowest_day": tbl.forecast.idxmin().strftime("%Y-%m-%d"), "lowest_day_forecast": _clean(tbl.forecast.min())}


# ----------------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def root():
    return {"service": "Sales Forecasting API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["system"])
def health():
    """Liveness/readiness probe used by Docker and Render."""
    ok = "model" in STATE
    body = {"status": "ok" if ok else "degraded", "model_loaded": ok}
    if ok:
        body.update(model_version=STATE["meta"]["model_version"], model_name=STATE["meta"]["model_name"])
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/model-info", tags=["model"])
def model_info():
    """Model version, features, accuracy metrics, uncertainty bands and feature importance."""
    _require_model(); m = STATE["meta"]
    keys = ["model_version", "trained_at", "data_hash", "model_name", "model_params", "feature_set", "use_lags", "features",
            "history_start", "history_end", "forecast_start", "forecast_end", "n_training_days", "validation_metrics", "test_metrics",
            "best_baseline", "intervals", "feature_importance", "n_models_compared", "inference_seconds_90d", "horizon_days", "window_errors"]
    return {k: m[k] for k in keys if k in m}


@app.get("/leaderboard", tags=["model"])
def leaderboard():
    """Top models from the model-comparison step of the notebook."""
    _require_model()
    return {"n_models_compared": STATE["meta"]["n_models_compared"], "top": STATE["meta"]["leaderboard"]}


@app.get("/forecast", tags=["forecast"])
def get_forecast(days: int = Query(90, ge=1, le=MAX_HORIZON, description=f"Number of days to forecast (1–{MAX_HORIZON})")):
    """Daily forecast for the next `days` days after the last day in the data, with a 5–95 % error band."""
    _require_model(); tbl = STATE["table"].iloc[:days]
    return {"model_version": STATE["meta"]["model_version"], "last_actual_date": STATE["meta"]["history_end"],
            "summary": _summary(tbl), "forecast": _rows(tbl)}


@app.get("/forecast/range", tags=["forecast"])
def get_forecast_range(start: date = Query(..., description="YYYY-MM-DD"), end: date = Query(..., description="YYYY-MM-DD")):
    """Forecast for a custom date range inside the forecast horizon."""
    _require_model(); t = STATE["table"]
    if end < start:
        raise HTTPException(status_code=422, detail="`end` must not be before `start`.")
    lo, hi = t.index.min().date(), t.index.max().date()
    if start < lo or end > hi:
        raise HTTPException(status_code=422, detail=f"Dates must be between {lo} and {hi} (the forecast horizon).")
    tbl = t.loc[str(start):str(end)]
    return {"summary": _summary(tbl), "forecast": _rows(tbl)}


@app.get("/forecast/date", tags=["forecast"])
def get_forecast_date(day: date = Query(..., alias="date", description="YYYY-MM-DD")):
    """Expected sales for ONE day. Past days return the actual recorded sales."""
    _require_model(); t, h = STATE["table"], STATE["history"]; ts = pd.Timestamp(day)
    if ts in h.index:
        return {"date": str(day), "weekday": ts.day_name(), "type": "actual", "sales": _clean(h.loc[ts])}
    if ts in t.index:
        r = t.loc[ts]
        return {"date": str(day), "weekday": ts.day_name(), "type": "forecast", "forecast": _clean(r.forecast),
                "lower": _clean(r.lower), "upper": _clean(r.upper), "days_ahead": int((ts - h.index.max()).days)}
    raise HTTPException(status_code=422, detail=f"Date must be between {h.index.min().date()} and {t.index.max().date()}.")


@app.get("/history", tags=["data"])
def get_history(days: Optional[int] = Query(None, ge=1, le=5000, description="Only the last N days (default: all)")):
    """Historical daily sales used to train the model."""
    _require_model(); h = STATE["history"]; h = h.iloc[-days:] if days else h
    return {"start": str(h.index.min().date()), "end": str(h.index.max().date()),
            "data": [{"date": d.strftime("%Y-%m-%d"), "sales": _clean(v)} for d, v in h.items()]}
