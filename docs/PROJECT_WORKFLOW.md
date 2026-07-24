# Comprehensive Project Workflow & System Documentation
## Enterprise Scalable LLM Fine-Tuning Platform

---

## 1. System Overview & Architecture

The **Enterprise Scalable LLM Fine-Tuning Platform** is a production-grade, end-to-end web application and API engine designed to manage the complete lifecycle of Large Language Model (LLM) fine-tuning.

### Architecture Diagram

```
 +-------------------------------------------------------------------------+
 |                     Black & Crimson Next.js Frontend                    |
 | (Dashboard, Berry UI Layout, Analytics Cards, Training Studio & Logs)   |
 +------------------------------------+------------------------------------+
                                      |
                               REST / WebSockets
                                      |
 +------------------------------------v------------------------------------+
 |                            FastAPI Backend Server                        |
 |                         (Port 9090 / Cloud Port)                        |
 +-------+-------------------+--------------------+-------------------+----+
         |                   |                    |                   |
  +------v------+     +------v------+      +------v------+     +------v------+
  | Auth & User |     | Datasets &  |      | Training &  |     | Evaluation  |
  | Management  |     | Cleaners    |      | Background  |     | & Inference |
  | (JWT Auth)  |     | (JSONL/CSV) |      | Workers     |     | Engine      |
  +------+------+     +------+------+      +------+------+     +------+------+
         |                   |                    |                   |
 +-------v-------------------v--------------------v-------------------v----+
 |                            SQLite / PostgreSQL DB                       |
 |                (Users, Datasets, Jobs, Registry, Metrics)                |
 +-------------------------------------------------------------------------+
```

---

## 2. End-to-End Pipeline Workflow

### Step 1: Authentication & User Management
- **Security**: PBKDF2 with SHA-256 password hashing and HMAC-SHA256 JWT tokens.
- **Endpoints**:
  - `POST /api/v1/auth/register`: Register user credentials.
  - `POST /api/v1/auth/login`: Authenticate and receive `access_token`.
  - `GET /api/v1/auth/me`: Retrieve active profile data.

### Step 2: Dataset Upload & Processing
- **Supported Formats**: `.jsonl`, `.csv`, `.json`, `.parquet`.
- **Validation**: Verifies JSON syntax and calculates sample counts and file sizes.
- **Preprocessing Engine**:
  - **Text Cleaning**: Unicode normalization, URL removal, HTML tag stripping, whitespace trimming.
  - **Deduplication**: Exact hash hashing & MinHash LSH deduplication.
- **Endpoints**:
  - `POST /api/v1/datasets/upload`: Ingest dataset files.
  - `POST /api/v1/datasets/{id}/preprocess`: Trigger text cleaning and deduplication.

### Step 3: Fine-Tuning Job Execution
- **Methods**: SFT (Supervised Fine-Tuning), LoRA, QLoRA (4-bit NormalFloat), DPO (Direct Preference Optimization).
- **Background Worker**: Asynchronous background thread executes epoch loss decay curves, tracking step counts, learning rate schedules, and VRAM memory consumption.
- **Endpoints**:
  - `POST /api/v1/training/launch`: Launch training workload.
  - `GET /api/v1/training/jobs`: List training jobs.
  - `POST /api/v1/training/jobs/{id}/stop`: Gracefully stop active job.
  - `/ws/telemetry`: Real-time WebSocket streaming of step progress and loss metrics.

### Step 4: Model Registration & PEFT Adapter Merging
- **Model Registry**: Auto-registers trained checkpoints with version numbers (`v1.0.X`).
- **PEFT Merge**: Merges low-rank adapter weights back into the base LLM for standalone deployment.
- **Endpoints**:
  - `POST /api/v1/models/register`: Register model checkpoint manually or automatically.
  - `POST /api/v1/models/{id}/merge`: Merge adapter weights into full model artifact.

### Step 5: Model Evaluation & Benchmarking
- **Metrics**: Perplexity (PPL), BLEU score, ROUGE-1, ROUGE-2, ROUGE-L, Exact Match (EM).
- **Sample Comparison**: Generates side-by-side responses comparing base model output vs. fine-tuned model output.
- **Endpoints**:
  - `POST /api/v1/evaluations/run`: Trigger evaluation benchmark.
  - `GET /api/v1/evaluations/{id}`: View detailed metric breakdowns.

### Step 6: One-Click Deployment & OpenAI API-Compatible Inference
- **Deployment**: Instantiates production endpoints with generated API keys (`sk_live_...`).
- **OpenAI Interface**: Full compatibility with standard OpenAI API clients.
- **Endpoints**:
  - `POST /api/v1/deployments`: Deploy registered model checkpoint.
  - `POST /v1/chat/completions`: Standard OpenAI Chat Completion endpoint.
  - `POST /v1/completions`: Standard OpenAI Text Completion endpoint.
  - `GET /v1/models`: List active deployed models.

### Step 7: Telemetry & Monitoring
- **Hardware Metrics**: Tracks real-time CPU, RAM, GPU utilization, VRAM usage, and compute cost accumulation.
- **Structured Logs**: Execution logs for auditing and debugging.
- **Endpoints**:
  - `GET /api/v1/monitoring/metrics`: Get cluster hardware metrics.
  - `GET /api/v1/monitoring/logs`: Fetch recent system execution logs.

---

## 3. Comprehensive Testing Strategy

The repository includes a complete multi-tier test suite using `pytest`:

1. **Unit Tests** (`tests/data_tests/`, `tests/training_tests/`):
   - Text cleaner normalization and strip validation.
   - Exact and MinHash deduplication ratio calculation.
   - Training hyperparameter resolution.
2. **Integration & Functional Tests** (`tests/test_sanity_and_functional.py`, `tests/integration_tests/`):
   - Model adapter merging.
   - Deployment lifecycle (creation, listing, termination).
   - Structured logging and OpenAI compatibility routes.
3. **End-to-End Workflow Tests** (`tests/test_e2e_workflow.py`):
   - Executes full pipeline: Auth -> Upload -> Preprocess -> Launch -> Checkpoint -> Evaluate -> Chat Inference -> Telemetry.

---

## 4. Render Deployment Instructions

### Method A: Automated Deployment via `render.yaml`
1. Connect your GitHub repository (`Kocherlasuhith12/scalable-llm-finetuning-pipeline`) to Render.
2. Render will automatically detect `render.yaml` and provision a Web Service:
   - **Environment**: Python 3.10+
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
3. Click **Apply / Deploy**.

### Method B: Manual Web Service Setup on Render
1. In Render Dashboard, click **New +** -> **Web Service**.
2. Connect repository `Kocherlasuhith12/scalable-llm-finetuning-pipeline`.
3. Set configuration:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables:
   - `DATABASE_URL`: `sqlite:///./platform.db`
   - `ENVIRONMENT`: `production`
   - `ALLOW_MOCK_TRAINING`: `true`
   - `SECRET_KEY`: `your-random-secret-key`
5. Click **Create Web Service**.

Once deployed, Render provides a public URL (e.g., `https://enterprise-llm-finetuning-platform.onrender.com`). Both the interactive Next.js dashboard UI and OpenAPI docs (`/docs`) are served live.
