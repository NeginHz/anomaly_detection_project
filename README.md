# IoT Anomaly Detection System (Real-Time Stream + REST API + Docker)

## Project Overview
This project implements a **real-time anomaly detection system** for an **IoT-enabled smart factory**.  
Sensors measure **temperature**, **humidity**, and **sound levels** during machine operations.  
A simple **Isolation Forest model** detects abnormal patterns and warns when anomalies occur.

The system is designed to be:
- **Monitorable** → includes `/health`, `/metrics`, and `/history` endpoints  
- **Maintainable** → training and serving are separated  
- **Scalable & adaptable** → easily containerized with Docker Compose  
- **Stream-based** → real-time simulated sensor input and live plotting  

---

## System Architecture

```
         ┌────────────────────┐
         │ make_baseline.py   │ → Generate baseline healthy data
         └──────────┬─────────┘
                    │
           ┌────────▼────────┐
           │   train.py      │ → Train IsolationForest model
           │  (baseline.csv) │
           └────────┬────────┘
                    │
             Artifacts (.pkl, .txt)
                    │
           ┌────────▼────────┐
           │   serve.py      │ → REST API (Flask)
           │ /health /predict /reload /metrics /history
           └────────┬────────┘
                    │
   ┌────────────────┴────────────────┐
   │                                 │
┌──▼────────────┐           ┌────────▼────────┐
│ sensor_sim.py │ → Stream  │ realtime_plot.py│
│ (send data)   │           │  (live chart)   │
└───────────────┘           └─────────────────┘
```

---

## Folder Structure
```
anomaly_detection_project/
│
├─ make_baseline.py                 ← Generate synthetic healthy baseline data
├─ README.md
├─ requirements.txt
├─ docker-compose.yml
├─ Dockerfile
│
├─ data/
│  ├─ baseline.csv                  ← Healthy training data
│  └─ sensor_log.csv                ← Real-time streaming log
│
├─ models/
│  ├─ model.pkl                     ← Trained IsolationForest model
│  ├─ scaler.pkl                    ← StandardScaler
│  └─ threshold.txt                 ← Quantile threshold (e.g., 0.11087)
│
├─ src/
│  ├─ train.py                      ← Train model on baseline data
│  └─ serve.py                      ← REST API for predictions
│
└─ scripts/
   ├─ sensor_sim.py                 ← Simulate IoT sensor stream
   └─ realtime_plot.py              ← Live plot of anomaly scores
```

---

## Installation (Local - Optional)

### 1️ Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# OR
.\.venv\Scripts\activate    # Windows
```

### 2️ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️ Run locally (optional)
```bash
# Step 1: Generate healthy baseline data
python make_baseline.py

# Step 2: Train the model
python -m src.train --baseline_path data/baseline.csv --model_dir models --quantile 0.99

# Step 3: Start the API service
python -m src.serve

# Step 4: Start the simulator in a separate terminal
python scripts/sensor_sim.py

# (Optional) View live anomaly scores
python scripts/realtime_plot.py
```

---

## Run with Docker (Recommended)

### 1️ Build & start
```bash
docker compose up -d --build
```

This starts:
- **api** → Flask REST API at `http://localhost:8000`
- **simulator** → continuously sends sensor readings to the API

Mounted volumes (persistent data):
- `./data` ↔ `/app/data`
- `./models` ↔ `/app/models`

---

### 2️ Quick checks
```bash
curl http://localhost:8000/health
curl http://localhost:8000/history
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"   -d '{"temperature":95,"humidity":20,"sound":120}'
curl http://localhost:8000/metrics
```

---

### 3️ Logs
```bash
docker logs -f anomaly_api
docker logs -f anomaly_sensor_sim
```

---

### 4️ Stop containers
```bash
docker compose down
```

---

### 5️ Rebuild after code changes
```bash
docker compose up -d --build
```

---

### 6️ Optional: increase stream rate or load
```bash
# Faster streaming (every 1 second)
# Edit docker-compose.yml → SLEEP_SEC=1

# Multiple sensor simulators (load testing)
docker compose up -d --scale simulator=3
```

---

## Key Endpoints Summary

| Endpoint | Method | Purpose | Returns |
|-----------|--------|----------|----------|
| `/health` | GET | Checks model readiness | `{"status":"ok"}` or `{"status":"unhealthy"}` |
| `/predict` | POST | Predict anomaly score | `{"anomaly_score":…, "is_anomaly":…, "threshold":…}` |
| `/reload` | POST | Reload model from disk | `{"status":"reloaded"}` |
| `/metrics` | GET | Prometheus metrics for monitoring | Text-formatted metrics |
| `/history?limit=N` | GET | Last N anomalies from `sensor_log.csv` | `{count, limit, items:[{timestamp, temperature, humidity, sound, anomaly_score, is_anomaly}]}` |

---

## Example Behavior

| Input (t, h, s) | anomaly_score | is_anomaly |
|-----------------|---------------|------------|
| (45, 50, 60)    | ~0.05         | 0          |
| (120, 5, 140)   | ~0.24         | 1          |

Low score → normal  
High score (above threshold) → anomaly detected

---

##  Design Features

| Feature | Description |
|----------|-------------|
| **Anomaly Detection Model** | IsolationForest trained on healthy baseline data |
| **Threshold** | 99th percentile of training anomaly scores |
| **Architecture** | Modular (training, serving, streaming separated) |
| **Streaming Support** | Continuous synthetic IoT data stream |
| **API Design** | RESTful (Flask) with `/predict`, `/history`, `/metrics` |
| **Health Check** | `/health` verifies artifacts and readiness |
| **Monitoring** | `/metrics` exposes Prometheus metrics (latency, counts) |
| **Reload Mechanism** | `/reload` loads updated model without restart |
| **Dockerized Deployment** | Fully containerized with `docker-compose` |
| **Persistence** | Shared data/models via Docker volumes |

---

##  Summary

**Everything works end-to-end:**
- Model trained and serialized  
- REST API containerized and live  
- Real-time data simulation running in Docker  
- Visualization via `realtime_plot.py`  
- Monitoring and anomaly metrics via `/metrics`  
- Recent anomalies retrievable via `/history`  
- Simple one-command deployment with Docker Compose  

This project demonstrates a **complete MLOps-style prototype** for **real-time IoT anomaly detection**, designed for maintainability, observability, and scalability.

---

*Author:*  
**Negin Hezarjaribi**  
Applied Artificial Intelligence Student – *Model to Product*  
*“From data to decisions, in real time.”*
