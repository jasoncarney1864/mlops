# MLOps Foundations: California Housing API (FastAPI + Docker)

This project trains a simple regression model using scikit-learn and serves predictions with FastAPI. It’s designed for WSL2 on Windows.

## 1) Setup (WSL2 Ubuntu)

```bash
# In your WSL2 terminal
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train the model

```bash
python src/train.py
# Outputs an RMSE and saves model to model/model.joblib
```

## 3) Run the API locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Visit: http://localhost:8000/docs for Swagger UI

### Test a prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3,
    "HouseAge": 41.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

## 4) Build and run with Docker

```bash
# Build image
docker build -t housing-api:latest .

# Run container
docker run --rm -p 8000:8000 housing-api:latest
```

Then open http://localhost:8000/docs.

## Makefile (optional)

```bash
# setup venv and install deps
make setup

# train the model
make train

# run api locally
make run-api

# docker build/run
make docker-build
make docker-run
```

## Notes
- Model artifact is saved to `model/model.joblib`. Ensure you run training before building the Docker image so the model is included.
- The dataset used is scikit-learn’s California Housing (no external credentials required).