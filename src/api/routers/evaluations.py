from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import EvaluationJob
from src.services.evaluation_service import run_evaluation_job

router = APIRouter(prefix="/api/v1/evaluations", tags=["Evaluation Studio"])

class EvaluationRunRequest(BaseModel):
    model_id: int
    dataset_id: Optional[int] = 1

@router.post("/run")
def trigger_evaluation(req: EvaluationRunRequest, db: Session = Depends(get_db)):
    try:
        job = run_evaluation_job(req.model_id, req.dataset_id or 1, db)
        return job
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
def list_evaluations(db: Session = Depends(get_db)):
    evals = db.query(EvaluationJob).order_by(EvaluationJob.id.desc()).all()
    return evals

@router.get("/{eval_id}")
def get_evaluation_details(eval_id: int, db: Session = Depends(get_db)):
    eval_job = db.query(EvaluationJob).filter(EvaluationJob.id == eval_id).first()
    if not eval_job:
        raise HTTPException(status_code=404, detail=f"Evaluation {eval_id} not found.")
    return eval_job
