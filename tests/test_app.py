from fastapi.testclient import TestClient
from app.app_sentiment import app


client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Sentiment Analysis Classifier API" in response.text

def test_predict_sentiment():
    response = client.post(
        "/predict",
        json={"text": "I love taking long walks in the park!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["negative", "neutral", "positive"]
    assert 0.0 <= data["confidence"] <= 1.0