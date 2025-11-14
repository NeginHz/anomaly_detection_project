# make_baseline.py
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

# Ensure data directory exists
Path("data").mkdir(parents=True, exist_ok=True)

# Number of healthy samples
n = 1000

# Create synthetic healthy sensor data
ts = pd.date_range(dt.datetime.utcnow(), periods=n, freq="3S")
temp = np.random.normal(45, 5, n)      # Temperature around 45°C ±5
hum  = np.random.normal(50, 7, n)      # Humidity around 50% ±7
snd  = np.random.normal(60, 6, n)      # Sound around 60 dB ±6

# Combine into a DataFrame
df = pd.DataFrame({
    "timestamp": ts.astype(str),
    "temperature": temp,
    "humidity": hum,
    "sound": snd
})

# Save to CSV
df.to_csv("data/baseline.csv", index=False)
print("baseline.csv created in data/ directory.")
