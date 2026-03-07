# Scalable LLM Fine-tuning Pipeline

End-to-end pipeline for fine-tuning large language models with distributed training, experiment tracking, and automated deployment.

**To use:** Download this entire folder (e.g. zip "Scalable LLM Fine-tuning Pipeline" or clone). All code, configs, and scripts are inside this single folder.
 
## Tech Stack

| Component | Technology |
|-----------|------------|
| Training | PyTorch, Hugging Face Transformers, DeepSpeed |
| Distributed | Ray, Horovod |
| Experiment Tracking | Weights & Biases / MLflow |
| Orchestration | Airflow / Prefect |
| Data Processing | Apache Spark / Dask |
| Storage | S3 / GCS / Azure Blob |
| Compute | AWS SageMaker / GCP AI Platform / Azure ML |

## Project Structure

This folder is the full project. Download the entire folder to get everything.

```
Scalable LLM Fine-tuning Pipeline/   # this folder
├── src/                    # Core pipeline code
│   ├── data/               # Data collection, processing, validation
│   ├── training/           # Trainers, configs, callbacks
│   ├── evaluation/         # Metrics, benchmarks, analysis
│   ├── deployment/         # Model conversion, quantization, API
│   ├── monitoring/         # Training & resource monitoring
│   └── utils/              # Distributed utils, checkpoint manager
├── workflows/              # DAGs and orchestration tasks
├── configs/                # YAML configuration files
├── experiments/            # Experiment logs and artifacts
├── scripts/                # Launch and utility scripts
├── tests/                  # Unit and integration tests
└── docker/                 # Container definitions
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp configs/base_config.yaml configs/local_config.yaml
# Edit local_config.yaml with your storage, compute, and API keys
```

### 3. Run Data Preparation

```bash
export PYTHONPATH="$PWD"  # from repo root
python -m workflows.dags.data_preparation --config configs/base_config.yaml
```

### 4. Launch Distributed Training

```bash
./scripts/launch_distributed_training.sh --config configs/lora_config.yaml
```

### 5. Evaluate Checkpoints

```bash
python scripts/evaluate_checkpoints.py --checkpoint-dir experiments/run_001
```

### 6. DPO (Preference) Alignment

After SFT, align the model to human preferences using (prompt, chosen, rejected) triplets:

```bash
# Preference data: one JSONL line per example
# {"prompt": "...", "chosen": "good response", "rejected": "bad response"}
cp data/preferences.example.jsonl data/processed/preferences.jsonl  # or use your own

python scripts/run_dpo_training.py --config configs/dpo_config.yaml --data-path data/processed/preferences.jsonl --output-dir outputs/dpo
```

Optional: train a reward model first for scoring or future PPO:

```bash
python scripts/run_dpo_training.py --config configs/dpo_config.yaml --train-reward-model
```

## Key Features

- **Data Pipeline**: Multi-source collection, distributed preprocessing, quality scoring, DVC versioning
- **Training**: Multi-GPU, LoRA/QLoRA, gradient checkpointing, mixed precision, checkpoint management
- **DPO / RLHF**: Direct Preference Optimization (and optional reward model) for alignment from preference data
- **Experiments**: Hyperparameter search (Optuna/Ray Tune), metric logging, cost estimation
- **Evaluation**: Benchmarks, custom metrics, statistical testing, visualization
- **Deployment**: Quantization (INT8/INT4), ONNX/TensorRT, API generation, load testing

## Advanced

- **DPO** – Implemented: preference dataset, DPO trainer (TRL), reward model trainer, config and pipeline
- Curriculum learning pipeline
- Synthetic data generation
- Continual learning with forgetting prevention
- Model compression

## License

MIT
