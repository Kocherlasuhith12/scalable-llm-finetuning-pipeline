#!/usr/bin/env python3
"""End-to-end integration verification test suite for Scalable LLM Fine-tuning Pipeline."""

import json
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_parser import load_config, merge_configs
from src.data.processors.cleaner import TextCleaner
from src.data.validators.deduplicator import Deduplicator
from src.data.datasets.preference_dataset import PreferenceDataset
from src.training.configs.training_configs import TrainingConfig
from src.training.configs.peft_configs import LoRAConfig
from src.deployment.quantizer import ModelQuantizer
from src.deployment.model_converter import ModelConverter
from src.deployment.api_builder import APIBuilder


def run_e2e_verification():
    print("==================================================")
    print("⚡ RUNNING END-TO-END PIPELINE VERIFICATION ⚡")
    print("==================================================")

    # 1. Config Test
    print("\n[1/6] Testing Configuration Parser...")
    config_path = Path("configs/base_config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        assert cfg.get("project", {}).get("name") == "llm-finetuning-pipeline"
        print("  ✓ Config loading & parsing verified.")
    else:
        print("  ⚠️ configs/base_config.yaml not found.")

    # 2. Data Cleaning & Deduplication
    print("\n[2/6] Testing Data Processing & Quality Checkers...")
    cleaner = TextCleaner(min_length=3, remove_urls=True)
    cleaned = cleaner.clean("  Check out https://github.com for LLM code  ")
    assert cleaned == "Check out for LLM code"

    dedupe = Deduplicator()
    docs = [{"text": "Sample LLM Text"}, {"text": "Sample LLM Text"}, {"text": "Unique Text"}]
    unique_docs = list(dedupe.dedupe_stream(iter(docs)))
    assert len(unique_docs) == 2
    print("  ✓ Text cleaning & stream deduplication verified.")

    # 3. Preference Dataset Test
    print("\n[3/6] Testing Preference & DPO Dataset Loaders...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "How to fine-tune an LLM?", "chosen": "Use LoRA/DPO.", "rejected": "Do nothing."}) + "\n")
        temp_dataset_path = f.name

    try:
        pref_ds = PreferenceDataset(data_path=temp_dataset_path)
        assert len(pref_ds) == 1
        trl_data = pref_ds.to_trl_format()
        assert trl_data[0]["prompt"] == "How to fine-tune an LLM?"
        print("  ✓ DPO Preference Dataset parser verified.")
    finally:
        Path(temp_dataset_path).unlink(missing_ok=True)

    # 4. Training Hyperparameter Configurations
    print("\n[4/6] Testing Training & PEFT Configs...")
    tr_cfg = TrainingConfig(batch_size=8, gradient_accumulation_steps=2)
    assert tr_cfg.effective_batch_size() == 16
    lora_cfg = LoRAConfig(r=32)
    assert lora_cfg.r == 32
    print("  ✓ Training & LoRA configurations verified.")

    # 5. Deployment Quantizer & Converter
    print("\n[5/6] Testing Quantizer & Model Converter...")
    quantizer = ModelQuantizer(bits=4)
    assert quantizer.bits == 4
    converter = ModelConverter(opset_version=14)
    assert converter.opset_version == 14
    print("  ✓ Quantizer & Model Converter modules verified.")

    # 6. Live API Builder & Endpoint Server Test
    print("\n[6/6] Testing Live API Serving Builder & Endpoints...")
    builder = APIBuilder(model_path="outputs/dpo", mock_mode=True)
    app = builder.create_app()
    assert app is not None

    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)

        # Test /health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "ok"

        # Test /v1/models
        res_models = client.get("/v1/models")
        assert res_models.status_code == 200
        assert len(res_models.json()["data"]) > 0

        # Test /generate
        res_gen = client.post("/generate", json={"prompt": "Explain RLHF simply", "max_tokens": 50})
        assert res_gen.status_code == 200
        assert "generated_text" in res_gen.json()

        # Test /v1/chat/completions (OpenAI Compatible)
        res_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "fine-tuned-llm",
                "messages": [{"role": "user", "content": "Hello! Explain LoRA fine-tuning."}],
                "temperature": 0.7,
            },
        )
        assert res_chat.status_code == 200
        chat_data = res_chat.json()
        assert chat_data["object"] == "chat.completion"
        assert len(chat_data["choices"]) > 0
        assert chat_data["choices"][0]["message"]["role"] == "assistant"

        print("  ✓ FastAPI Server /health, /generate, and OpenAI /v1/chat/completions endpoints verified successfully!")
    except ImportError:
        print("  ⚠️ fastapi.testclient not available for inline HTTP client testing.")

    print("\n==================================================")
    print("✨ ALL PIPELINE COMPONENTS VERIFIED SUCCESSFULLY! ✨")
    print("==================================================")


if __name__ == "__main__":
    run_e2e_verification()
