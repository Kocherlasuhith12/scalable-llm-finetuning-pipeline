import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_user_profile_sanity():
    """Sanity check: Login and fetch current user profile (/api/v1/auth/me)."""
    login_res = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    user_data = me_res.json()
    assert user_data["username"] == "admin"
    assert user_data["role"] == "admin"

def test_model_registration_and_adapter_merge_functional():
    """Functional test: Register model checkpoint and execute adapter merge."""
    register_payload = {
        "name": "llama3-med-adapter-v1",
        "version": "v1.0.0",
        "base_model": "meta-llama/Llama-3.2-1B",
        "training_job_id": 1,
        "artifact_path": "outputs/job_1_lora",
        "quantization": "fp16"
    }
    reg_res = client.post("/api/v1/models/register", json=register_payload)
    assert reg_res.status_code == 200
    model_data = reg_res.json()
    model_id = model_data["id"]
    assert model_data["status"] == "registered"
    
    # Execute adapter merge
    merge_res = client.post(f"/api/v1/models/{model_id}/merge")
    assert merge_res.status_code == 200
    merged_data = merge_res.json()
    assert merged_data["status"] == "merged"
    assert f"merged_model_id_{model_id}" in merged_data["artifact_path"]

def test_deployment_lifecycle_functional():
    """Functional test: Create deployment endpoint, list deployments, and terminate."""
    # First register a model
    reg_res = client.post("/api/v1/models/register", json={
        "name": "deployable-llama",
        "version": "v1.0.1",
        "base_model": "meta-llama/Llama-3.2-1B",
        "artifact_path": "outputs/job_2"
    })
    model_id = reg_res.json()["id"]
    
    # Deploy model
    deploy_res = client.post("/api/v1/deployments", json={"model_id": model_id, "name": "production-llm-endpoint"})
    assert deploy_res.status_code == 200
    dep_data = deploy_res.json()
    dep_id = dep_data["id"]
    assert dep_data["status"] == "active"
    assert dep_data["api_key"].startswith("sk_live_")
    
    # List deployments
    list_res = client.get("/api/v1/deployments")
    assert list_res.status_code == 200
    deployments = list_res.json()
    assert any(d["id"] == dep_id for d in deployments)
    
    # Terminate deployment
    term_res = client.delete(f"/api/v1/deployments/{dep_id}")
    assert term_res.status_code == 200
    assert term_res.json()["status"] == "success"

def test_monitoring_logs_functional():
    """Functional test: Retrieve structured execution logs."""
    res = client.get("/api/v1/monitoring/logs?limit=10")
    assert res.status_code == 200
    logs = res.json()
    assert isinstance(logs, list)
    assert len(logs) > 0
    assert "timestamp" in logs[0]
    assert "level" in logs[0]
    assert "message" in logs[0]

def test_text_completions_legacy_functional():
    """Functional test: OpenAI legacy /v1/completions endpoint."""
    res = client.post("/v1/completions", json={"prompt": "Explain gradient descent", "model": "meta-llama/Llama-3.2-1B"})
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

def test_training_job_stop_functional():
    """Functional test: Launch fine-tuning job and issue stop command."""
    # Launch job
    launch_res = client.post("/api/v1/training/launch", json={
        "name": "stoppable-job",
        "base_model": "meta-llama/Llama-3.2-1B",
        "dataset_id": 1,
        "method": "lora",
        "epochs": 5
    })
    assert launch_res.status_code in [200, 201]
    job_id = launch_res.json()["id"]
    
    # Stop job
    stop_res = client.post(f"/api/v1/training/jobs/{job_id}/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"
