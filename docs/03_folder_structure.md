# 03. Folder Structure

```
scalable-llm-finetuning-pipeline/
├── docs/                      # Architectural documentation (01-18)
├── configs/                   # YAML configurations for LoRA, QLoRA, DPO, Base
├── docker/                    # Dockerfiles for train, eval, and serve
├── render.yaml                # Render cloud deployment blueprint
├── Dockerfile                 # Root production container setup
├── docker-compose.yml         # Container orchestration manifest
├── pyproject.toml             # Project metadata & tools config
├── requirements.txt           # Python dependencies
├── scripts/                   # CLI execution & utility scripts
│   ├── serve.py               # Serving entry point
│   ├── run_dpo_training.py    # DPO training script
│   ├── evaluate_checkpoints.py# Evaluation runner
│   ├── merge_adapters.py      # PEFT adapter merger
│   ├── test_pipeline_e2e.py   # E2E pipeline test
│   └── deploy_live.sh         # Production deploy helper
├── src/                       # Main source package
│   ├── api/                   # FastAPI routes & app entry point
│   │   ├── routers/           # Auth, datasets, training, evals, models, etc.
│   │   └── main.py            # FastAPI main app setup
│   ├── auth/                  # JWT auth, hashing & route protection
│   ├── data/                  # Data collectors, processors, validators
│   ├── database/              # SQLAlchemy database ORM models & session
│   ├── deployment/            # API builder, quantizer, model converter
│   ├── evaluation/            # Evaluators, metrics, analysis
│   ├── monitoring/            # Resource tracker, training monitor, costs
│   ├── services/              # Business logic & background workers
│   ├── static/                # Frontend Web Dashboard (HTML, CSS, JS)
│   ├── training/              # Trainers (Base, LoRA, QLoRA, DPO), callbacks
│   └── utils/                 # Config parser, distributed utils, checkpoints
└── tests/                     # Unit and integration test suite
```
