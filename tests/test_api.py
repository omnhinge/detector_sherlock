from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "Detector Sherlock"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_verify_empty():
    response = client.post("/verify", json={"response": "Short."})
    assert response.status_code == 200
    data = response.json()
    assert data["claims_count"] == 0
    assert data["summary"]["reliability_score"] == 0.0
