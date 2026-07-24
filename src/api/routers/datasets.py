from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import Dataset, PreprocessingJob, User
from src.auth.security import get_current_user_optional
from src.services.dataset_service import process_file_upload, run_dataset_preprocessing

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])

class PreprocessRequest(BaseModel):
    min_length: int = 5
    normalize_unicode: bool = True
    remove_urls: bool = True
    strip_html: bool = True
    dedup_threshold: float = 0.9

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    user_id = user.id if user else None
    dataset = process_file_upload(file.filename, content, db, user_id)
    return dataset

@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.id.desc()).all()
    return datasets

@router.get("/{dataset_id}")
def get_dataset_details(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    prep_jobs = db.query(PreprocessingJob).filter(PreprocessingJob.dataset_id == dataset_id).all()
    return {
        "dataset": dataset,
        "preprocessing_history": prep_jobs
    }

@router.post("/{dataset_id}/preprocess")
def preprocess_dataset_endpoint(
    dataset_id: int,
    req: PreprocessRequest,
    db: Session = Depends(get_db)
):
    try:
        rules = req.dict()
        job = run_dataset_preprocessing(dataset_id, rules, db)
        return job
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    db.delete(dataset)
    db.commit()
    return {"status": "success", "message": f"Dataset {dataset_id} deleted."}
