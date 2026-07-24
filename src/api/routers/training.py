from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import TrainingJob, User
from src.auth.security import get_current_user_optional
from src.services.training_service import create_and_launch_training_job, stop_training_job

router = APIRouter(prefix="/api/v1/training", tags=["Training Studio"])

class TrainingLaunchRequest(BaseModel):
    name: str
    base_model: str = "meta-llama/Llama-3.2-1B"
    dataset_id: int
    method: str = "lora"  # sft, lora, qlora, dpo
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    quantization: str = "fp16"

@router.post("/launch")
def launch_training(
    req: TrainingLaunchRequest,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    user_id = user.id if user else None
    hyperparams = {
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "learning_rate": req.learning_rate,
        "lora_r": req.lora_r,
        "lora_alpha": req.lora_alpha,
        "target_modules": req.target_modules,
        "quantization": req.quantization,
        "total_steps": req.epochs * 25
    }
    job = create_and_launch_training_job(
        name=req.name,
        base_model=req.base_model,
        dataset_id=req.dataset_id,
        method=req.method,
        hyperparameters=hyperparams,
        db=db,
        owner_id=user_id
    )
    return job

@router.get("/jobs")
def list_training_jobs(db: Session = Depends(get_db)):
    jobs = db.query(TrainingJob).order_by(TrainingJob.id.desc()).all()
    return jobs

@router.get("/jobs/{job_id}")
def get_training_job_details(job_id: int, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found.")
    return job

@router.post("/jobs/{job_id}/stop")
def stop_job_endpoint(job_id: int, db: Session = Depends(get_db)):
    try:
        job = stop_training_job(job_id, db)
        return job
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
