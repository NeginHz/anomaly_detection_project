#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple REST API to serve an anomaly detection model trained on baseline data.
Endpoints:
  - GET  /health    -> health/readiness check (deep: tiny inference)
  - POST /predict   -> predict anomaly score and label from JSON payload
  - GET  /metrics   -> basic Prometheus-style metrics (optional)
  - POST /reload    -> reload artifacts from disk without restarting
  - GET  /history   -> return recent anomaly events from data/sensor_log.csv

Run:
  python -m src.serve

Environment variables (optional):
  API_HOST=0.0.0.0
  API_PORT=8000
  MODEL_DIR=models
"""

import os
import csv
import logging
import pickle
import time
from typing import Tuple
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# Optional metrics (won't break if not installed)
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

# ----------------------------
# Configuration (kept inline for simplicity)
# ----------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MODEL_DIR = os.getenv("MODEL_DIR", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

LOG_PATH = Path("data/sensor_log.csv")  # for /history endpoint

# Simple validation ranges (adjust to your domain if needed)
RANGE_TEMPERATURE = (-20.0, 150.0)  # Celsius
RANGE_HUMIDITY    = (0.0, 100.0)    # %
RANGE_SOUND       = (0.0, 160.0)    # dB

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Global artifacts (safe state)
# ----------------------------
model = None
scaler = None
THRESHOLD = None

def try_load_artifacts() -> bool:
    """Try to load model, scaler, and threshold from disk. Never crash."""
    global model, scaler, THRESHOLD
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(THRESHOLD_PATH, "r") as f:
            THRESHOLD = float(f.read().strip())
        logging.info("Artifacts loaded OK (threshold=%.8f)", THRESHOLD)
        return True
    except Exception as e:
        model = None
        scaler = None
        THRESHOLD = None
        logging.warning("Artifacts NOT loaded: %s", e)
        return False

# ----------------------------
# Metrics (optional)
# ----------------------------
if PROM_AVAILABLE:
    REQ_COUNT = Counter("api_requests_total", "Total number of API requests", ["endpoint", "method", "status"])
    REQ_LAT   = Histogram("api_request_latency_seconds", "API request latency (seconds)", ["endpoint"])
else:
    REQ_COUNT = None
    REQ_LAT   = None

def observe_request(endpoint: str, start_time: float, status: int):
    """Record metrics if prometheus_client is available."""
    if PROM_AVAILABLE:
        REQ_COUNT.labels(endpoint=endpoint, method=request.method, status=status).inc()
        REQ_LAT.labels(endpoint=endpoint).observe(time.time() - start_time)

# ----------------------------
# App
# ----------------------------
app = Flask(__name__)
CORS(app)  # keep open in dev; restrict in production

# Try to load artifacts at startup (non-fatal)
_ = try_load_artifacts()

# ----------------------------
# Helpers
# ----------------------------
def parse_and_validate(payload: dict) -> Tuple[float, float, float]:
    """
    Validate incoming JSON. Ensure keys exist and values are numeric within a reasonable range.
    Raise 400 on invalid input.
    """
    required = ["temperature", "humidity", "sound"]
    for k in required:
        if k not in payload:
            abort(400, description=f"Missing field: {k}")

    try:
        t = float(payload["temperature"])
        h = float(payload["humidity"])
        s = float(payload["sound"])
    except Exception:
        abort(400, description="Fields must be numeric (temperature, humidity, sound).")

    # Range checking to avoid garbage inputs
    if not (RANGE_TEMPERATURE[0] <= t <= RANGE_TEMPERATURE[1]):
        abort(400, description=f"temperature out of range {RANGE_TEMPERATURE}")
    if not (RANGE_HUMIDITY[0] <= h <= RANGE_HUMIDITY[1]):
        abort(400, description=f"humidity out of range {RANGE_HUMIDITY}")
    if not (RANGE_SOUND[0] <= s <= RANGE_SOUND[1]):
        abort(400, description=f"sound out of range {RANGE_SOUND}")

    return t, h, s

def compute_score(x_row: np.ndarray) -> float:
    """
    Compute anomaly score for a single 1x3 feature row.
    Uses negative decision_function so higher => more anomalous.
    """
    x_scaled = scaler.transform(x_row)
    score = -model.decision_function(x_scaled)[0]
    return float(score)

def read_recent_anomalies(limit: int = 10):
    """
    Read last `limit` anomaly rows (is_anomaly==1) from data/sensor_log.csv.
    Returns a list of dicts in reverse-chronological order (latest first).

    Expected CSV header:
      timestamp,temperature,humidity,sound,anomaly_score,is_anomaly
    """
    if not LOG_PATH.exists():
        return []

    # Sliding window of last anomalies
    buf = deque(maxlen=limit)

    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    is_anom = int(row.get("is_anomaly", "0"))
                except ValueError:
                    is_anom = 0
                if is_anom == 1:
                    try:
                        item = {
                            "timestamp": row.get("timestamp"),
                            "temperature": float(row.get("temperature")) if row.get("temperature") not in (None, "",) else None,
                            "humidity": float(row.get("humidity")) if row.get("humidity") not in (None, "",) else None,
                            "sound": float(row.get("sound")) if row.get("sound") not in (None, "",) else None,
                            "anomaly_score": float(row.get("anomaly_score")) if row.get("anomaly_score") not in (None, "",) else None,
                            "is_anomaly": 1
                        }
                    except Exception:
                        # Skip malformed numeric rows
                        continue
                    buf.append(item)
    except Exception as e:
        # If file is being written concurrently, return whatever we have
        logging.warning("history read warning: %s", e)

    # Latest first
    return list(buf)[::-1]

# ----------------------------
# Endpoints
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    """
    Health/readiness check.
    - Returns 200 if artifacts are loaded and a tiny inference works.
    - Returns 503 if not ready.
    """
    # Artifacts present?
    if (model is None) or (scaler is None) or (THRESHOLD is None):
        return jsonify({"status": "unhealthy", "detail": "artifacts not loaded"}), 503

    # Tiny end-to-end check
    try:
        dummy = np.array([[45.0, 50.0, 60.0]], dtype=float)
        _xs = scaler.transform(dummy)
        _ = -model.decision_function(_xs)[0]
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "detail": str(e)}), 503

@app.route("/reload", methods=["POST"])
def reload_artifacts():
    """
    Reload artifacts from disk without restarting the server.
    - 200 if loaded
    - 503 if still missing/broken
    """
    ok = try_load_artifacts()
    if ok:
        return jsonify({"status": "reloaded"}), 200
    return jsonify({"status": "failed", "detail": "artifacts not found or invalid"}), 503

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    status = 200
    try:
        # Ensure artifacts are available
        if (model is None) or (scaler is None) or (THRESHOLD is None):
            status = 503
            return jsonify({"error": "model not loaded"}), 503

        payload = request.get_json(silent=True)
        if payload is None:
            status = 400
            abort(400, description="Invalid or missing JSON body.")

        t, h, s = parse_and_validate(payload)
        x = np.array([[t, h, s]], dtype=float)
        score = compute_score(x)
        is_anomaly = int(score > THRESHOLD)

        return jsonify({
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "threshold": THRESHOLD
        }), 200

    except Exception as e:
        # Flask abort() already sets proper status codes; others => 500
        if hasattr(e, "code") and isinstance(e.code, int):
            status = e.code
            raise
        else:
            logging.exception("Unhandled error in /predict")
            status = 500
            return jsonify({"error": "internal server error"}), 500
    finally:
        observe_request("/predict", start, status)

@app.route("/metrics", methods=["GET"])
def metrics():
    start = time.time()
    try:
        if not PROM_AVAILABLE:
            return jsonify({"detail": "prometheus_client not installed"}), 200
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
    finally:
        observe_request("/metrics", start, 200)

@app.route("/history", methods=["GET"])
def history():
    """
    Return recent anomaly events from sensor_log.csv.
    Query params:
      - limit (int, optional, default=10) → number of rows to return (clamped to [1, 500])
    Response:
      {
        "count": <int>,
        "limit": <int>,
        "items": [
          {timestamp, temperature, humidity, sound, anomaly_score, is_anomaly}
        ]
      }
    """
    start = time.time()
    status = 200
    try:
        limit_q = request.args.get("limit", default="10")
        try:
            limit = int(limit_q)
            if limit < 1: limit = 1
            if limit > 500: limit = 500
        except Exception:
            limit = 10

        items = read_recent_anomalies(limit=limit)
        return jsonify({
            "count": len(items),
            "limit": limit,
            "items": items
        }), 200
    except Exception:
        logging.exception("Unhandled error in /history")
        status = 500
        return jsonify({"error": "internal server error"}), 500
    finally:
        observe_request("/history", start, status)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    logging.info("Starting API on %s:%d", API_HOST, API_PORT)
    app.run(host=API_HOST, port=API_PORT)  # debug=False for a cleaner log
