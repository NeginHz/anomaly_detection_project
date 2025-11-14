#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realtime plot of anomaly scores from data/sensor_log.csv.
Shows the last N points and a horizontal line at the stored threshold.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOG_PATH = "data/sensor_log.csv"
THR_PATH = "models/threshold.txt"
WINDOW_N = 300   # show the last N points

def load_threshold():
    if Path(THR_PATH).exists():
        return float(Path(THR_PATH).read_text().strip())
    return None

def load_tail():
    if not Path(LOG_PATH).exists():
        return pd.DataFrame(columns=["timestamp","anomaly_score"])
    df = pd.read_csv(LOG_PATH)
    # Keep only what we need
    cols = ["timestamp", "anomaly_score"]
    df = df[cols].tail(WINDOW_N).copy()
    # Safe numeric conversion
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    df = df.dropna()
    return df

def animate(_):
    df = load_tail()
    thr = load_threshold()
    ax.clear()
    ax.set_title("Realtime Anomaly Score")
    ax.set_xlabel("Time (last points)")
    ax.set_ylabel("Anomaly Score")

    if len(df) > 0:
        y = df["anomaly_score"].values
        x = np.arange(len(y))
        ax.plot(x, y, marker="o", linewidth=1)
        if thr is not None:
            ax.axhline(thr, linestyle="--")
        ax.grid(True)

fig, ax = plt.subplots()
ani = FuncAnimation(fig, animate, interval=1000)  # update every 1s

if __name__ == "__main__":
    plt.show()
