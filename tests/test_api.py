import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "version" in data

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["active_model"] == "MobileNetV3"
    assert len(data["classes"]) == 7
    assert "model_comparison" in data

def test_predict_endpoint_valid_image(client):
    # Create valid synthetic image in memory
    img = Image.new("RGB", (224, 224), color=(100, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_waste.jpg", buf, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert "prediction" in data
    assert "recommendation" in data
    assert "explainability" in data
    assert data["prediction"]["class"] in [
        "plastic", "paper", "glass", "metal", "organic", "textile", "e_waste"
    ]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    assert data["recommendation"]["environmental_priority"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_endpoint_invalid_extension(client):
    files = {"file": ("script.sh", b"echo hello", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]

def test_history_and_statistics_flow(client):
    # Predict image first
    img = Image.new("RGB", (224, 224), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("metal_can.png", buf, "image/png")}
    pred_res = client.post("/predict", files=files)
    assert pred_res.status_code == 200
    pred_id = pred_res.json()["id"]

    # Check history
    hist_res = client.get("/history?limit=10")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert len(hist) > 0
    assert any(h["id"] == pred_id for h in hist)

    # Check statistics
    stats_res = client.get("/statistics")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_predictions"] >= 1

    # Check feedback
    fb_res = client.post(
        "/feedback",
        json={"prediction_id": pred_id, "user_feedback": "correct"}
    )
    assert fb_res.status_code == 200
    assert fb_res.json()["success"] is True
