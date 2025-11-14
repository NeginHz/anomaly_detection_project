#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate sensor readings and send them to the model API.
Append results to data/sensor_log.csv for later visualization.
"""

import os
import time
import json
import random
from pathlib import Path
from datetime import datetime, timezone

import requests

API_URL   = os.getenv("API_URL", "http://localhost:8000/predict")
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "3"))
LOG_PATH  = os.getenv("LOG_PATH", "data/sensor_log.csv")

# Ensure data directory exists
Path("data").mkdir(parents=True, exist_ok=True)

# Write CSV header if file does not exist
if not Path(LOG_PATH).exists():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("timestamp,temperature,humidity,sound,anomaly_score,is_anomaly\n")

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def generate_sample():
    """
    Generate a mostly healthy reading with occasional anomalies.
    """
    # Base healthy ranges (tweak to match your baseline)
    t = random.gauss(45, 5)   # temperature
    h = random.gauss(50, 7)   # humidity
    s = random.gauss(60, 6)   # sound

    # Occasionally inject anomalies (~5% chance)
    if random.random() < 0.05:
        t += random.choice([+40, +60])   # spike
        h += random.choice([-30, +30])   # drop/spike
        s += random.choice([+40, +60])   # spike

    return round(t, 2), round(h, 2), round(s, 2)

def main():
    print(f"Sending sensor readings to: {API_URL} every {SLEEP_SEC}s")
    while True:
        t, h, s = generate_sample()
        payload = {"temperature": t, "humidity": h, "sound": s}

        try:
            r = requests.post(API_URL, json=payload, timeout=3)
            if r.status_code == 200:
                resp = r.json()
                score = resp.get("anomaly_score", "")
                label = resp.get("is_anomaly", "")
                # Append to CSV
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(f"{now_utc_iso()},{t},{h},{s},{score},{label}\n")
                print(f"[OK] t={t} h={h} s={s} -> score={score} label={label}")
            else:
                print(f"[WARN] HTTP {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()
