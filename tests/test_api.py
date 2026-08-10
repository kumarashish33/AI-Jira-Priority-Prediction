from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "Application crashes after login"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] in [
        "Highest",
        "High",
        "Medium",
        "Low",
    ]

    assert 0 <= data["confidence"] <= 1
