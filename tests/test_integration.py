import requests

BASE = "http://localhost:7860"

def test_root():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200

def test_predict():
    r = requests.post(f"{BASE}/predict", json={"text": "I love this!"})
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] in ["negative", "neutral", "positive"]
    assert 0.8 <= data["confidence"] <= 1.0

def test_mlflow_proxy():
    r = requests.get(f"{BASE}/mlflow/")
    assert r.status_code == 200

def test_predict_to_logs():
    r = requests.post(f"{BASE}/predict", json={"text": "This is a wonderful day!"})
    predicted = r.json()
    logs = requests.get(f"{BASE}/logs", params={"last_n": 1}).json()
    latest = logs["recent_predictions"][0]
    assert latest["label"] == predicted["sentiment"]
