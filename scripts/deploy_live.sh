#!/usr/bin/env bash
# Production Live Server Deployment Script

set -e

echo "=== 🚀 Deploying Scalable LLM Fine-Tuning API to Live Server ==="

# Check if Docker is available
if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
    echo "🐳 Docker and Docker Compose detected. Launching container service..."
    docker-compose down || true
    docker-compose up -d --build
    echo "✅ Service successfully started in background via Docker Compose on port 8000!"
    echo "🌐 Test health endpoint: curl http://localhost:8000/health"
    exit 0
fi

# Fallback: Run directly with Python virtual environment
echo "⚠️ Docker not detected. Falling back to native Python process serving..."

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt

echo "Starting model API server in background..."
nohup python scripts/serve.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &

echo "✅ Server started with PID $! (logs saved to server.log)"
echo "🌐 Test health endpoint: curl http://localhost:8000/health"
