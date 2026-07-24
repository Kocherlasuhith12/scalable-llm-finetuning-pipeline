import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_full_e2e_pipeline_lifecycle():
    """
    End-to-End System Test:
    1. Health & Readiness Check
    2. User Authentication (JWT Token)
    3. Dataset File Upload & Format Validation
    4. Dataset Processing & Preprocessing
    5. Fine-Tuning Job Launch (QLoRA) & Workload Status
    6. Model Checkpoint Registration & Listing
    7. Evaluation Benchmark Trigger (BLEU, ROUGE, PPL)
    8. OpenAI Compatible Chat Completion Inference
    9. Real-time GPU Telemetry Monitoring
    """
    # Step 1: Health & Readiness Check
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    ready_res = client.get("/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["status"] == "ready"

    # Step 2: Authentication (Login as seed admin user)
    auth_res = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert auth_res.status_code == 200
    token_data = auth_res.json()
    assert "access_token" in token_data
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}

    # Step 3: Dataset Ingestion via Multipart File Upload
    file_content = b'{"instruction": "What is QLoRA?", "response": "QLoRA is Quantized Low-Rank Adaptation."}\n'
    files = {"file": ("e2e_medical_qa.jsonl", file_content, "application/jsonl")}
    dataset_res = client.post("/api/v1/datasets/upload", files=files, headers=headers)
    assert dataset_res.status_code in [200, 201]
    dataset = dataset_res.json()
    assert "id" in dataset
    dataset_id = dataset["id"]

    # Step 4: Dataset Preprocessing Execution
    preprocess_payload = {
        "min_length": 5,
        "normalize_unicode": True,
        "remove_urls": True,
        "strip_html": True,
        "dedup_threshold": 0.9
    }
    prep_res = client.post(f"/api/v1/datasets/{dataset_id}/preprocess", json=preprocess_payload, headers=headers)
    assert prep_res.status_code == 200
    assert prep_res.json()["status"].lower() == "completed"

    # Step 5: Fine-Tuning Workload Launch (QLoRA)
    training_payload = {
        "name": "e2e-qlora-llama3-med-job",
        "base_model": "meta-llama/Llama-3.2-1B",
        "dataset_id": dataset_id,
        "method": "qlora",
        "learning_rate": 0.0002,
        "epochs": 3,
        "batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32
    }
    job_res = client.post("/api/v1/training/launch", json=training_payload, headers=headers)
    assert job_res.status_code in [200, 201]
    job = job_res.json()
    assert job["name"] == "e2e-qlora-llama3-med-job"
    assert job["method"] == "qlora"

    # Step 6: Model Checkpoint Listing
    models_res = client.get("/api/v1/models", headers=headers)
    assert models_res.status_code == 200
    models_list = models_res.json()
    assert isinstance(models_list, list)

    # Step 7: Evaluation Benchmark Trigger
    eval_payload = {"model_id": 1, "dataset_id": dataset_id}
    eval_res = client.post("/api/v1/evaluations/run", json=eval_payload, headers=headers)
    assert eval_res.status_code in [200, 201]
    eval_data = eval_res.json()
    assert "metrics" in eval_data
    assert "perplexity" in eval_data["metrics"]
    assert "bleu" in eval_data["metrics"]

    # Step 8: OpenAI Compatible Chat Inference Execution
    chat_payload = {
        "model": "meta-llama/Llama-3.2-1B",
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain QLoRA fine-tuning in simple terms."}
        ]
    }
    chat_res = client.post("/v1/chat/completions", json=chat_payload, headers=headers)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "choices" in chat_data
    assert len(chat_data["choices"]) > 0
    assert "message" in chat_data["choices"][0]

    # Step 9: Telemetry Monitoring Endpoint
    metrics_res = client.get("/api/v1/monitoring/metrics", headers=headers)
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert "cpu_percent" in metrics
    assert "gpu_percent" in metrics
    assert "vram_used_mb" in metrics
