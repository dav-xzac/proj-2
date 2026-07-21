import sys
from unittest.mock import MagicMock, patch

# Mock the modules before app loading for fast testing for push on develop branch
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

from fastapi.testclient import TestClient
from app_sentiment import app

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200

# the function analyze_sentiment is also mocked to prevent impossible operation on the mocked object
@patch("app_sentiment.analyze_sentiment", return_value=("positive", 0.95))
def test_predict(mock_analyze):
    r = client.post("/predict", json={"text": "great!"})
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] in ["negative", "neutral", "positive"]
    assert data["confidence"] == 0.95

def test_predict_missing_field():
    r = client.post("/predict", json={})
    assert r.status_code == 422

def test_logs_invalid_last_n():
    r = client.get("/logs", params={"last_n": 0})
    assert r.status_code == 422


