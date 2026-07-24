from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.database.database import get_db
from src.database.models import DeploymentEndpoint, ModelRegistry
from src.services.deployment_service import generate_chat_completion

router = APIRouter(prefix="/v1", tags=["OpenAI Inference API"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "meta-llama/Llama-3.2-1B"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256

@router.post("/chat/completions")
def create_chat_completion_endpoint(
    req: ChatCompletionRequest,
    db: Session = Depends(get_db)
):
    try:
        messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
        res = generate_chat_completion(
            model_name=req.model,
            messages=messages_dicts,
            temperature=req.temperature or 0.7,
            max_tokens=req.max_tokens or 256,
            db=db
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/completions")
def create_text_completion_endpoint(
    req: Dict[str, Any],
    db: Session = Depends(get_db)
):
    model_name = req.get("model", "meta-llama/Llama-3.2-1B")
    prompt = req.get("prompt", "")
    messages = [{"role": "user", "content": prompt}]
    res = generate_chat_completion(model_name, messages, db=db)
    return res

@router.get("/models")
def list_available_openai_models(db: Session = Depends(get_db)):
    deployments = db.query(DeploymentEndpoint).filter(DeploymentEndpoint.status == "active").all()
    model_entries = []
    
    if not deployments:
        model_entries.append({
            "id": "meta-llama/Llama-3.2-1B",
            "object": "model",
            "created": 1700000000,
            "owned_by": "platform"
        })
    else:
        for dep in deployments:
            model_name = dep.model.name if dep.model else dep.name
            model_entries.append({
                "id": model_name,
                "object": "model",
                "created": int(dep.created_at.timestamp()) if dep.created_at else 1700000000,
                "owned_by": "custom-fine-tuned"
            })
            
    return {
        "object": "list",
        "data": model_entries
    }
