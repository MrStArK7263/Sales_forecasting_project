# ---- Sales Forecasting API ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts \
    MAX_HORIZON=180 \
    LOG_LEVEL=INFO

WORKDIR /app

# libgomp1 is needed by LightGBM / XGBoost / scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forecast_core.py main.py ./
COPY artifacts ./artifacts

# run as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fs http://localhost:${PORT:-8000}/health || exit 1

# Render injects $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
