# test/test_app.py
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_prediction_endpoint():
    response = client.post("/predict-json", json={"email": "Click here to reset your PayPal password."})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()