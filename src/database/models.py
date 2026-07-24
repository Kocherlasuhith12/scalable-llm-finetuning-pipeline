from datetime import datetime
import json
from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from src.database.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="owner")
    training_jobs = relationship("TrainingJob", back_populates="owner")

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # jsonl, csv, parquet
    sample_count = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    status = Column(String(30), default="uploaded")  # uploaded, preprocessed, error
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="datasets")
    preprocessing_jobs = relationship("PreprocessingJob", back_populates="dataset")
    training_jobs = relationship("TrainingJob", back_populates="dataset")

class PreprocessingJob(Base):
    __tablename__ = "preprocessing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    status = Column(String(30), default="pending")  # pending, processing, completed, failed
    cleaning_rules = Column(JSON, default=dict)
    deduplication_ratio = Column(Float, default=0.0)
    processed_count = Column(Integer, default=0)
    output_file_path = Column(String(255), nullable=True)
    error_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="preprocessing_jobs")

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    base_model = Column(String(100), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    method = Column(String(30), nullable=False)  # sft, lora, qlora, dpo
    hyperparameters = Column(JSON, default=dict)
    status = Column(String(30), default="pending")  # pending, running, completed, failed, stopped
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_loss = Column(Float, default=0.0)
    metrics_history = Column(JSON, default=list)
    output_dir = Column(String(255), nullable=True)
    error_log = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="training_jobs")
    owner = relationship("User", back_populates="training_jobs")
    models = relationship("ModelRegistry", back_populates="training_job")

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    version = Column(String(30), nullable=False)
    base_model = Column(String(100), nullable=False)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    artifact_path = Column(String(255), nullable=False)
    quantization = Column(String(20), default="none")
    eval_metrics = Column(JSON, default=dict)
    status = Column(String(30), default="registered")  # registered, merged, deployed
    created_at = Column(DateTime, default=datetime.utcnow)

    training_job = relationship("TrainingJob", back_populates="models")
    evaluations = relationship("EvaluationJob", back_populates="model")
    deployments = relationship("DeploymentEndpoint", back_populates="model")

class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    status = Column(String(30), default="pending")  # pending, running, completed, failed
    metrics = Column(JSON, default=dict)  # perplexity, bleu, rouge_1, rouge_2, rouge_l
    sample_outputs = Column(JSON, default=list)
    error_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRegistry", back_populates="evaluations")

class DeploymentEndpoint(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    model_id = Column(Integer, ForeignKey("model_registry.id"), nullable=False)
    endpoint_url = Column(String(255), nullable=True)
    status = Column(String(30), default="active")  # active, stopped, error
    requests_handled = Column(Integer, default=0)
    avg_latency_ms = Column(Float, default=0.0)
    api_key = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRegistry", back_populates="deployments")

class SystemMetric(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cpu_percent = Column(Float, default=0.0)
    ram_percent = Column(Float, default=0.0)
    gpu_percent = Column(Float, default=0.0)
    vram_used_mb = Column(Float, default=0.0)
    cost_accumulated = Column(Float, default=0.0)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    details = Column(Text, nullable=True)
