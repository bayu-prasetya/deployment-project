from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_predict():
    res = client.post("/predict", json={"age": 30, "income": 8000, "gender": "M"})
    assert res.status_code == 200
    assert "prediction" in res.json()