from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_and_readiness_endpoints():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_auth_register_and_login():
    reg_payload = {
        "username": "api_test_user_unique",
        "email": "api_test_unique@example.com",
        "password": "Password123"
    }
    res = client.post("/api/v1/auth/register", json=reg_payload)
    assert res.status_code in [200, 400]
    
    login_payload = {
        "username": "api_test_user_unique" if res.status_code == 200 else "admin",
        "password": "Password123" if res.status_code == 200 else "admin123"
    }
    res = client.post("/api/v1/auth/login", json=login_payload)
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data

def test_openai_chat_completions_endpoint():
    payload = {
        "model": "meta-llama/Llama-3.2-1B",
        "messages": [{"role": "user", "content": "What is LoRA fine-tuning?"}],
        "temperature": 0.7
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "LoRA" in data["choices"][0]["message"]["content"] or "Low-Rank Adaptation" in data["choices"][0]["message"]["content"]

def test_monitoring_telemetry_endpoint():
    res = client.get("/api/v1/monitoring/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "cpu_percent" in data
    assert "ram_percent" in data
    assert "gpu_percent" in data
