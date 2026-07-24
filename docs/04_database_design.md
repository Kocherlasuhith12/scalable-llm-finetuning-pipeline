# 04. Database Design

## Database Entities & Schema Definitions

### 1. `users` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `username` (VARCHAR(50), Unique, Not Null)
- `email` (VARCHAR(120), Unique, Not Null)
- `hashed_password` (VARCHAR(255), Not Null)
- `role` (VARCHAR(20), Default: 'user')
- `is_active` (BOOLEAN, Default: True)
- `created_at` (DATETIME, Default: UTC Now)

### 2. `datasets` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `name` (VARCHAR(100), Not Null)
- `file_path` (VARCHAR(255), Not Null)
- `file_type` (VARCHAR(20), Not Null: 'jsonl', 'csv', 'parquet')
- `sample_count` (INTEGER, Default: 0)
- `size_bytes` (INTEGER, Default: 0)
- `status` (VARCHAR(30), Default: 'uploaded')
- `owner_id` (INTEGER, Foreign Key -> users.id)
- `created_at` (DATETIME, Default: UTC Now)

### 3. `preprocessing_jobs` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `dataset_id` (INTEGER, Foreign Key -> datasets.id)
- `status` (VARCHAR(30), Default: 'pending')
- `cleaning_rules` (JSON)
- `deduplication_ratio` (FLOAT, Default: 0.0)
- `processed_count` (INTEGER, Default: 0)
- `output_file_path` (VARCHAR(255))
- `created_at` (DATETIME, Default: UTC Now)

### 4. `training_jobs` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `name` (VARCHAR(100), Not Null)
- `base_model` (VARCHAR(100), Not Null)
- `dataset_id` (INTEGER, Foreign Key -> datasets.id)
- `method` (VARCHAR(30), Not Null: 'sft', 'lora', 'qlora', 'dpo')
- `hyperparameters` (JSON, Stores lr, epochs, lora_r, lora_alpha, batch_size, target_modules)
- `status` (VARCHAR(30), Default: 'pending': 'running', 'completed', 'failed', 'stopped')
- `current_step` (INTEGER, Default: 0)
- `total_steps` (INTEGER, Default: 0)
- `current_loss` (FLOAT, Default: 0.0)
- `metrics_history` (JSON)
- `output_dir` (VARCHAR(255))
- `owner_id` (INTEGER, Foreign Key -> users.id)
- `created_at` (DATETIME, Default: UTC Now)

### 5. `model_registry` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `name` (VARCHAR(100), Not Null)
- `version` (VARCHAR(30), Not Null)
- `base_model` (VARCHAR(100), Not Null)
- `training_job_id` (INTEGER, Foreign Key -> training_jobs.id)
- `artifact_path` (VARCHAR(255), Not Null)
- `quantization` (VARCHAR(20), Default: 'none')
- `eval_metrics` (JSON)
- `status` (VARCHAR(30), Default: 'registered')
- `created_at` (DATETIME, Default: UTC Now)

### 6. `evaluation_jobs` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `model_id` (INTEGER, Foreign Key -> model_registry.id)
- `dataset_id` (INTEGER, Foreign Key -> datasets.id)
- `status` (VARCHAR(30), Default: 'pending')
- `metrics` (JSON, Stores perplexity, bleu, rouge_1, rouge_2, rouge_l)
- `sample_outputs` (JSON)
- `created_at` (DATETIME, Default: UTC Now)

### 7. `deployments` Table
- `id` (INTEGER, Primary Key, Auto-increment)
- `name` (VARCHAR(100), Not Null)
- `model_id` (INTEGER, Foreign Key -> model_registry.id)
- `endpoint_url` (VARCHAR(255))
- `status` (VARCHAR(30), Default: 'active')
- `requests_handled` (INTEGER, Default: 0)
- `avg_latency_ms` (FLOAT, Default: 0.0)
- `api_key` (VARCHAR(100))
- `created_at` (DATETIME, Default: UTC Now)

### 8. `system_metrics` & `audit_logs` Tables
- Stores time-series server resource metrics (CPU, RAM, GPU) and historical security audit records.
