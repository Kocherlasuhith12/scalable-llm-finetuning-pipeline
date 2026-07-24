# 13. Deployment on Render & Cloud Platforms

## Render Blueprint (`render.yaml`)
- **Service Type**: Web Service (`env: python` or `docker`).
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `DATABASE_URL`: `sqlite:///./platform.db` (or PostgreSQL URL).
  - `SECRET_KEY`: Production secret key.
  - `ENVIRONMENT`: `production`.
  - `ALLOW_MOCK_TRAINING`: `true` (enables simulated GPU training execution in non-GPU cloud instances).

## Docker Container Deployment
Multi-stage build Dockerfile packaging FastAPI server, static dashboard UI, and execution engine into a single production container.
