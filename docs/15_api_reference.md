# 15. API Reference

## Authentication Endpoints
- `POST /api/v1/auth/register`: Register user (`username`, `email`, `password`).
- `POST /api/v1/auth/login`: Authenticate and obtain JWT `access_token`.
- `GET /api/v1/auth/me`: Get current authenticated user profile.

## Dataset Endpoints
- `POST /api/v1/datasets/upload`: Upload dataset file (`file`, `name`).
- `GET /api/v1/datasets`: List uploaded datasets.
- `GET /api/v1/datasets/{id}`: Dataset details & statistical summary.
- `POST /api/v1/datasets/{id}/preprocess`: Trigger cleaning & deduplication job.
- `DELETE /api/v1/datasets/{id}`: Delete dataset.

## Training Endpoints
- `POST /api/v1/training/launch`: Launch fine-tuning job (`name`, `base_model`, `dataset_id`, `method`, `hyperparameters`).
- `GET /api/v1/training/jobs`: List training jobs with current step/loss status.
- `GET /api/v1/training/jobs/{id}`: Detailed training job metrics & loss history.
- `POST /api/v1/training/jobs/{id}/stop`: Terminate active training job.

## Model Registry Endpoints
- `POST /api/v1/models/register`: Register trained checkpoint into model registry.
- `GET /api/v1/models`: List registered models.
- `POST /api/v1/models/{id}/merge`: Execute PEFT adapter merging into base model.
- `POST /api/v1/models/{id}/deploy`: Deploy model as an active inference endpoint.

## OpenAI-Compatible Inference Endpoints
- `POST /v1/chat/completions`: OpenAI standard chat completions (`model`, `messages`, `temperature`, `top_p`, `stream`).
- `POST /v1/completions`: Standard text completions.
- `GET /v1/models`: List available deployed models.

## Monitoring & WebSockets
- `GET /api/v1/monitoring/metrics`: Get live CPU, RAM, GPU, cost stats.
- `GET /api/v1/monitoring/logs`: Get filterable execution logs.
- `WS /ws/telemetry`: WebSockets live stream for hardware telemetry & active job metrics.
