from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import DeploymentEndpoint
from src.services.deployment_service import create_deployment

router = APIRouter(prefix="/api/v1/deployments", tags=["Deployments"])

class DeployModelRequest(BaseModel):
    model_id: int
    name: Optional[str] = None

@router.post("")
def deploy_model_endpoint(req: DeployModelRequest, db: Session = Depends(get_db)):
    try:
        deployment = create_deployment(req.model_id, req.name, db)
        return deployment
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
def list_deployments(db: Session = Depends(get_db)):
    deployments = db.query(DeploymentEndpoint).order_by(DeploymentEndpoint.id.desc()).all()
    return deployments

@router.delete("/{deployment_id}")
def terminate_deployment(deployment_id: int, db: Session = Depends(get_db)):
    deployment = db.query(DeploymentEndpoint).filter(DeploymentEndpoint.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found.")
    deployment.status = "stopped"
    db.commit()
    return {"status": "success", "message": f"Deployment {deployment_id} stopped."}
