import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.database.database import init_db, SessionLocal
from src.database.models import User
from src.auth.security import hash_password

from src.api.routers.auth import router as auth_router
from src.api.routers.datasets import router as datasets_router
from src.api.routers.training import router as training_router
from src.api.routers.evaluations import router as evaluations_router
from src.api.routers.models import router as models_router
from src.api.routers.deployments import router as deployments_router
from src.api.routers.inference import router as inference_router
from src.api.routers.monitoring import router as monitoring_router
from src.api.routers.websockets import router as websockets_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Scalable LLM Fine-Tuning Platform",
    description="Production-grade platform for LLM fine-tuning, evaluation, versioning, deployment, and monitoring.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure DB tables are initialized
init_db()

# Initialize Database and Seed Default User
@app.on_event("startup")
def startup_event():
    logger.info("Verifying database tables and seeding defaults...")
    init_db()
    
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            logger.info("Seeding default admin user (admin / admin123)...")
            admin_user = User(
                username="admin",
                email="admin@enterprise.ai",
                hashed_password=hash_password("admin123"),
                role="admin"
            )
            db.add(admin_user)
            db.commit()
    finally:
        db.close()

# Register Routers
app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(evaluations_router)
app.include_router(models_router)
app.include_router(deployments_router)
app.include_router(inference_router)
app.include_router(monitoring_router)
app.include_router(websockets_router)

# Mount Static Dashboard UI
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

_next_dir = os.path.join(static_dir, "_next")
if os.path.exists(_next_dir):
    app.mount("/_next", StaticFiles(directory=_next_dir), name="next")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({
        "name": "Enterprise Scalable LLM Fine-Tuning Platform API",
        "status": "online",
        "docs_url": "/docs"
    })

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "llm-finetuning-platform"}

@app.get("/ready")
def readiness_check():
    return {"status": "ready"}
