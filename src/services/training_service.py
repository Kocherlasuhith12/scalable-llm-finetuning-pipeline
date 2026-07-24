import os
import time
import math
import random
import json
import logging
import threading
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from src.database.database import SessionLocal
from src.database.models import TrainingJob, ModelRegistry

logger = logging.getLogger(__name__)

# Active background training threads
_active_training_threads: Dict[int, threading.Thread] = {}
_stop_signals: Dict[int, bool] = {}

# Active WebSocket subscribers for live progress updates
_ws_listeners: List[Any] = []

def register_ws_listener(listener_func):
    _ws_listeners.append(listener_func)

def unregister_ws_listener(listener_func):
    if listener_func in _ws_listeners:
        _ws_listeners.remove(listener_func)

def broadcast_training_update(data: dict):
    for listener in list(_ws_listeners):
        try:
            listener(data)
        except Exception:
            pass

def create_and_launch_training_job(
    name: str,
    base_model: str,
    dataset_id: int,
    method: str,
    hyperparameters: Dict[str, Any],
    db: Session,
    owner_id: Optional[int] = None
) -> TrainingJob:
    """Create training job record in DB and launch async training task."""
    epochs = int(hyperparameters.get("epochs", 3))
    batch_size = int(hyperparameters.get("batch_size", 4))
    total_steps = int(hyperparameters.get("total_steps", epochs * 25))
    
    output_dir = os.path.join("outputs", f"job_{int(time.time())}_{method}")
    os.makedirs(output_dir, exist_ok=True)
    
    job = TrainingJob(
        name=name,
        base_model=base_model,
        dataset_id=dataset_id,
        method=method,
        hyperparameters=hyperparameters,
        status="running",
        current_step=0,
        total_steps=total_steps,
        current_loss=2.5,
        metrics_history=[],
        output_dir=output_dir,
        owner_id=owner_id
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    job_id = job.id
    _stop_signals[job_id] = False
    
    thread = threading.Thread(
        target=_run_training_worker,
        args=(job_id, name, base_model, method, total_steps, hyperparameters, output_dir),
        daemon=True
    )
    _active_training_threads[job_id] = thread
    thread.start()
    
    return job

def stop_training_job(job_id: int, db: Session) -> TrainingJob:
    """Stop active training job."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise ValueError(f"Job {job_id} not found.")
        
    _stop_signals[job_id] = True
    job.status = "stopped"
    db.commit()
    db.refresh(job)
    return job

def _run_training_worker(
    job_id: int,
    job_name: str,
    base_model: str,
    method: str,
    total_steps: int,
    hyperparameters: dict,
    output_dir: str
):
    """Background worker executing training steps and updating progress."""
    db = SessionLocal()
    try:
        initial_loss = 2.8
        target_loss = 0.35 + random.uniform(0.05, 0.15)
        decay_rate = 3.5 / total_steps
        lr = float(hyperparameters.get("learning_rate", 2e-4))
        
        metrics_history = []
        
        for step in range(1, total_steps + 1):
            if _stop_signals.get(job_id, False):
                logger.info(f"Training job {job_id} stopped by user signal.")
                break
                
            time.sleep(0.4)  # Step delay for realistic streaming
            
            # Loss decay curve simulation with micro-fluctuations
            noise = random.uniform(-0.02, 0.02)
            current_loss = max(target_loss, round(initial_loss * math.exp(-decay_rate * step) + noise, 4))
            current_lr = round(lr * (1.0 - (step / total_steps) * 0.8), 6)
            gpu_mem_mb = round(3400 + math.sin(step / 5.0) * 200 + step * 2.5, 1)
            
            metric_entry = {
                "step": step,
                "loss": current_loss,
                "learning_rate": current_lr,
                "gpu_memory_mb": gpu_mem_mb,
                "epoch": round(step / (total_steps / float(hyperparameters.get("epochs", 3))), 2),
                "timestamp": time.strftime("%H:%M:%S")
            }
            metrics_history.append(metric_entry)
            
            # Update DB periodically or at completion
            if step % 2 == 0 or step == total_steps:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job:
                    job.current_step = step
                    job.current_loss = current_loss
                    job.metrics_history = metrics_history
                    db.commit()
                    
            # Broadcast to UI via WebSockets
            broadcast_training_update({
                "type": "training_progress",
                "job_id": job_id,
                "step": step,
                "total_steps": total_steps,
                "loss": current_loss,
                "learning_rate": current_lr,
                "gpu_memory_mb": gpu_mem_mb,
                "progress_pct": round((step / total_steps) * 100, 1)
            })
            
        # Finalize job
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job and not _stop_signals.get(job_id, False):
            job.status = "completed"
            job.current_step = total_steps
            db.commit()
            
            # Auto register trained model into registry
            version_str = f"v1.0.{job_id}"
            model_reg = ModelRegistry(
                name=f"{job_name}-checkpoint",
                version=version_str,
                base_model=base_model,
                training_job_id=job_id,
                artifact_path=output_dir,
                quantization=hyperparameters.get("quantization", "fp16"),
                eval_metrics={"final_training_loss": current_loss},
                status="registered"
            )
            db.add(model_reg)
            db.commit()
            
            broadcast_training_update({
                "type": "training_completed",
                "job_id": job_id,
                "final_loss": current_loss,
                "model_name": model_reg.name,
                "version": version_str
            })
            
    except Exception as e:
        logger.error(f"Error in training worker for job {job_id}: {e}", exc_info=True)
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_log = str(e)
            db.commit()
    finally:
        db.close()
        _active_training_threads.pop(job_id, None)
