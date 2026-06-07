import sys
from unittest.mock import MagicMock, patch

sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

@patch("app_sentiment.analyze_sentiment", return_value=("positive", 0.95))
def test_predict(mock_analyze):
    from fastapi.testclient import TestClient
    from app_sentiment import app
    client = TestClient(app)
    r = client.post("/predict", json={"text": "great!"})
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] in ["negative", "neutral", "positive"]
    assert 0.7 < data["confidence"] <= 1.0
  
@patch("app_sentiment.analyze_sentiment", return_value=("positive", 0.95))
def test_health(mock_analyze):
    from fastapi.testclient import TestClient
    from app_sentiment import app
    client = TestClient(app)
    assert client.get("/health").status_code == 200