from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import ModelRegistry
from src.services.deployment_service import merge_model_adapters

router = APIRouter(prefix="/api/v1/models", tags=["Model Registry"])

class ModelRegisterRequest(BaseModel):
    name: str
    version: str = "v1.0.0"
    base_model: str
    training_job_id: Optional[int] = None
    artifact_path: str
    quantization: str = "fp16"

@router.post("/register")
def register_model_checkpoint(req: ModelRegisterRequest, db: Session = Depends(get_db)):
    model = ModelRegistry(
        name=req.name,
        version=req.version,
        base_model=req.base_model,
        training_job_id=req.training_job_id,
        artifact_path=req.artifact_path,
        quantization=req.quantization,
        eval_metrics={},
        status="registered"
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

@router.get("")
def list_models(db: Session = Depends(get_db)):
    models = db.query(ModelRegistry).order_by(ModelRegistry.id.desc()).all()
    return models

@router.get("/{model_id}")
def get_model_details(model_id: int, db: Session = Depends(get_db)):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return model

@router.post("/{model_id}/merge")
def merge_adapters_endpoint(model_id: int, db: Session = Depends(get_db)):
    try:
        model = merge_model_adapters(model_id, db)
        return model
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
