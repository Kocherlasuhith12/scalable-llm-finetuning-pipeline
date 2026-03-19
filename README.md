<div align="center">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/DeepSpeed-0078D4?style=for-the-badge&logo=microsoft&logoColor=white"/>
<img src="https://img.shields.io/badge/Ray-028CF0?style=for-the-badge&logo=ray&logoColor=white"/>
<img src="https://img.shields.io/badge/AWS_SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/W%26B-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black"/>
<img src="https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# ⚡ Scalable LLM Fine-tuning Pipeline

**Production-grade, end-to-end pipeline for fine-tuning large language models —  
from raw data to aligned, deployed, and monitored models at scale.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [DPO Alignment](#-dpo-alignment-rlhf) · [Deployment](#-deployment) · [Tech Stack](#-tech-stack)

</div>

---

## 🧠 What Is This?

A fully modular, research-to-production LLM fine-tuning system that handles every stage of the ML lifecycle:

- **Data ingestion & quality filtering** at scale (Spark / Dask + DVC versioning)
- **Distributed SFT training** with LoRA / QLoRA, DeepSpeed ZeRO, and mixed precision (FP16/BF16)
- **Preference alignment** via DPO (Direct Preference Optimization) and optional Reward Modeling for RLHF
- **Automated hyperparameter search** with Optuna / Ray Tune
- **Full experiment tracking** with Weights & Biases and MLflow
- **Model compression & deployment** — INT4/INT8 quantization, ONNX/TensorRT export, REST API generation
- **Real-time training monitoring** with resource dashboards

> Built to take a raw dataset all the way to a fine-tuned, aligned, and deployed LLM — without stitching together 10 different repos.

---

## ✨ Features

| Category | Capabilities |
|---|---|
| 🗃️ **Data Pipeline** | Multi-source collection · Quality scoring · DVC versioning · Distributed preprocessing |
| 🏋️ **Training** | Multi-GPU · LoRA / QLoRA · Gradient checkpointing · Mixed precision · Checkpoint manager |
| 🎯 **Alignment** | DPO trainer (TRL) · Reward model training · RLHF-ready · Preference dataset handling |
| 🔬 **Experiments** | Optuna / Ray Tune HPO · Metric logging · Cost estimation · A/B comparison |
| 📊 **Evaluation** | Standard benchmarks · Custom metrics · Statistical testing · Visualization reports |
| 🚀 **Deployment** | INT4 / INT8 quantization · ONNX + TensorRT export · Auto API generation · Load testing |
| 🩺 **Monitoring** | Live training dashboards · GPU/CPU resource tracking · Drift detection |
| 🧪 **Advanced** | Curriculum learning · Synthetic data generation · Continual learning · Model compression |

---

## 🏗️ Architecture
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Scalable LLM Fine-tuning Pipeline                      │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┤
│  Data Layer  │   Training   │  Alignment   │  Evaluation  │   Deployment     │
│              │              │              │              │                  │
│ Spark / Dask │  DeepSpeed   │     DPO      │  Benchmarks  │  Quantization    │
│   Quality    │   ZeRO-3     │   Trainer    │   Metrics    │  ONNX/TensorRT   │
│   Scoring    │  LoRA/QLoRA  │    (TRL)     │   Reports    │  REST API        │
│  DVC Track   │  Ray/Horovod │  Reward Mdl  │   Viz Tools  │  Load Testing    │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴──────────────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
              Orchestrated by Airflow / Prefect
              Tracked by W&B / MLflow
              Stored on S3 / GCS / Azure Blob
```

---

## 📂 Project Structure
```
scalable-llm-finetuning-pipeline/
│
├── src/                          # Core pipeline modules
│   ├── data/                     # Collection, preprocessing, validation, quality scoring
│   ├── training/                 # Trainers, configs, distributed strategies, callbacks
│   ├── evaluation/               # Metrics, benchmarks, statistical testing, visualization
│   ├── deployment/               # Quantization, ONNX/TensorRT export, API server generation
│   ├── monitoring/               # Training monitor, resource tracker, dashboards
│   └── utils/                    # Distributed utils, checkpoint manager, helpers
│
├── workflows/                    # Orchestration
│   └── dags/                     # Airflow / Prefect DAGs
│
├── configs/                      # YAML configuration files
│   ├── base_config.yaml          # Base configuration template
│   ├── lora_config.yaml          # LoRA / QLoRA training config
│   └── dpo_config.yaml           # DPO / RLHF alignment config
│
├── experiments/                  # Experiment logs, artifacts, checkpoints
├── scripts/                      # Launch, evaluate, DPO, and utility scripts
├── tests/                        # Unit + integration tests
├── docker/                       # Dockerfiles and Compose files
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Kocherlasuhith12/scalable-llm-finetuning-pipeline.git
cd scalable-llm-finetuning-pipeline
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp configs/base_config.yaml configs/local_config.yaml
# Edit local_config.yaml — set storage paths, compute targets, and API keys
```

### 3. Run Data Preparation
```bash
export PYTHONPATH="$PWD"
python -m workflows.dags.data_preparation --config configs/base_config.yaml
```

### 4. Launch Distributed Training (SFT)
```bash
./scripts/launch_distributed_training.sh --config configs/lora_config.yaml
```

> Supports multi-GPU (single node) and multi-node (Ray / Horovod) out of the box.

### 5. Evaluate Checkpoints
```bash
python scripts/evaluate_checkpoints.py --checkpoint-dir experiments/run_001
```

---

## 🎯 DPO Alignment (RLHF)

After supervised fine-tuning, align your model to human preferences using **(prompt, chosen, rejected)** triplets.

**Preference data format** — one JSONL line per example:
```json
{"prompt": "Explain quantum entanglement simply.", "chosen": "Clear, helpful response...", "rejected": "Overly technical response..."}
```

**Run DPO training:**
```bash
cp data/preferences.example.jsonl data/processed/preferences.jsonl

python scripts/run_dpo_training.py \
  --config configs/dpo_config.yaml \
  --data-path data/processed/preferences.jsonl \
  --output-dir outputs/dpo
```

**Optional — Train a Reward Model first** (for scoring candidates or future PPO):
```bash
python scripts/run_dpo_training.py \
  --config configs/dpo_config.yaml \
  --train-reward-model
```

---

## 🔬 Experiment Tracking

All runs are automatically tracked with **Weights & Biases** and/or **MLflow**.

Each run logs:
- Training loss, validation loss, perplexity
- GPU/CPU utilization and memory usage
- Learning rate schedule and gradient norms
- Checkpoint artifacts and model diffs
- Estimated compute cost

Switch backends via `base_config.yaml`:
```yaml
experiment_tracking:
  backend: wandb   # or: mlflow
  project: llm-finetuning
```

---

## 🚀 Deployment

Export and serve your fine-tuned model in production:
```bash
# Quantize to INT8 or INT4
python src/deployment/quantize.py --model outputs/dpo --bits 8

# Export to ONNX / TensorRT
python src/deployment/export.py --format onnx --model outputs/dpo

# Auto-generate REST API
python src/deployment/serve.py --model outputs/dpo --port 8000
```

| Target | Options |
|---|---|
| Quantization | INT4, INT8 (bitsandbytes, GPTQ) |
| Export | ONNX, TensorRT |
| Cloud Compute | AWS SageMaker · GCP AI Platform · Azure ML |
| Storage | S3 · GCS · Azure Blob |

---

## 🧪 Advanced Capabilities

| Feature | Status |
|---|---|
| DPO / RLHF Alignment | ✅ Implemented |
| Curriculum Learning | ✅ Implemented |
| Synthetic Data Generation | ✅ Implemented |
| Continual Learning (anti-forgetting) | ✅ Implemented |
| Model Compression | ✅ Implemented |
| Hyperparameter Optimization (Optuna / Ray Tune) | ✅ Implemented |
| Reward Model Training | ✅ Implemented |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Training Framework | PyTorch · Hugging Face Transformers · TRL |
| Distributed Training | DeepSpeed ZeRO · Ray · Horovod |
| Experiment Tracking | Weights & Biases · MLflow |
| Orchestration | Apache Airflow · Prefect |
| Data Processing | Apache Spark · Dask |
| Storage | AWS S3 · GCP GCS · Azure Blob |
| Compute | AWS SageMaker · GCP AI Platform · Azure ML |
| Containerization | Docker · Docker Compose |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

---

<div align="center">

<br>

*Built with 🧠 by* &nbsp; [![Author](https://img.shields.io/badge/Kocherla_Koteswara_Suhith_Sravan_Babu-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Kocherlasuhith12)

<br>

> ⭐ If this project helped you, consider leaving a **star** — it means a lot. ⭐

<br>

</div>
