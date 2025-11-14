#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a simple anomaly detection model (IsolationForest) on a 'healthy' baseline dataset.
Outputs:
  - models/scaler.pkl
  - models/model.pkl
  - models/threshold.txt   (quantile-based threshold on training scores)

Run:
  python -m src.train \
    --baseline_path data/baseline.csv \
    --model_dir models \
    --quantile 0.99 \
    --random_state 42
"""

import os
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection model on baseline data.")
    parser.add_argument("--baseline_path", type=str, default="data/baseline.csv",
                        help="Path to baseline CSV (healthy-only records).")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save model artifacts.")
    parser.add_argument("--quantile", type=float, default=0.99,
                        help="Quantile for threshold selection (0 < q < 1).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

# ----------------------------
# Setup logging
# ----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

# ----------------------------
# Data loading & validation
# ----------------------------
def load_and_validate(baseline_path: str) -> pd.DataFrame:
    """
    Load baseline CSV and validate required columns.
    Expected columns: ['temperature', 'humidity', 'sound'] (timestamp is optional).
    Drops rows with missing values in the required columns.
    """
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    df = pd.read_csv(baseline_path)

    required_cols = ["temperature", "humidity", "sound"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in baseline: {missing}. "
                         f"Expected at least {required_cols}")

    # Keep only required features; drop rows with NaN in any required feature
    df = df[required_cols].copy()
    before = len(df)
    df = df.dropna()
    after = len(df)

    if after == 0:
        raise ValueError("All rows were dropped due to missing values in required features.")

    if after < before:
        logging.warning("Dropped %d rows with missing values in required features.", before - after)

    # Basic numeric sanity check (not strict; just avoids totally broken inputs)
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Baseline contains non-finite values (inf/-inf). Clean your data.")

    return df

# ----------------------------
# Training pipeline
# ----------------------------
def train_isolation_forest(X: np.ndarray, random_state: int) -> tuple[StandardScaler, IsolationForest]:
    """
    Fit StandardScaler and IsolationForest on the provided features.
    Returns fitted scaler and model.
    """
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # A very simple, robust choice for MVP
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=random_state
    ).fit(X_scaled)

    return scaler, model

def compute_scores(model: IsolationForest, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """
    Compute raw anomaly scores. We use negative decision_function so that
    higher values mean 'more anomalous'.
    """
    X_scaled = scaler.transform(X)
    raw = -model.decision_function(X_scaled)  # larger = more anomalous
    return raw

def pick_threshold(scores: np.ndarray, q: float) -> float:
    """
    Select a quantile-based threshold (e.g., 0.99) from training scores.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("Quantile must be in (0, 1).")
    return float(np.quantile(scores, q))

# ----------------------------
# Saving artifacts
# ----------------------------
def save_artifacts(model_dir: str, scaler: StandardScaler, model: IsolationForest, threshold: float):
    """
    Persist scaler, model, and threshold to disk.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path = os.path.join(model_dir, "model.pkl")
    thr_path = os.path.join(model_dir, "threshold.txt")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(thr_path, "w") as f:
        f.write(f"{threshold:.8f}")

    logging.info("Saved scaler -> %s", scaler_path)
    logging.info("Saved model  -> %s", model_path)
    logging.info("Saved threshold -> %s (quantile)", thr_path)

# ----------------------------
# Main
# ----------------------------
def main():
    setup_logging()
    args = parse_args()

    logging.info("Loading baseline from: %s", args.baseline_path)
    df = load_and_validate(args.baseline_path)

    # Extract features matrix
    X = df[["temperature", "humidity", "sound"]].to_numpy(dtype=float)
    logging.info("Baseline shape: %s rows x %s features", X.shape[0], X.shape[1])

    # Train
    logging.info("Training IsolationForest (random_state=%d)...", args.random_state)
    scaler, model = train_isolation_forest(X, args.random_state)

    # Scores & threshold
    scores = compute_scores(model, scaler, X)
    threshold = pick_threshold(scores, args.quantile)

    # Simple descriptive stats for reporting
    logging.info("Score stats on baseline | mean=%.4f std=%.4f min=%.4f p99=%.4f max=%.4f",
                 float(np.mean(scores)),
                 float(np.std(scores)),
                 float(np.min(scores)),
                 float(np.quantile(scores, 0.99)),
                 float(np.max(scores)))
    logging.info("Selected threshold (quantile=%.3f): %.6f", args.quantile, threshold)

    # Persist artifacts
    logging.info("Saving artifacts into: %s", args.model_dir)
    save_artifacts(args.model_dir, scaler, model, threshold)

    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()
