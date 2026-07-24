#!/usr/bin/env python3
"""Launch script for production model serving API server."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deployment.api_builder import APIBuilder


def main():
    parser = argparse.ArgumentParser(description="Serve Fine-tuned LLM via FastAPI REST API")
    parser.add_argument("--model", type=str, default="outputs/dpo", help="Path to fine-tuned model or adapter")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--mock", action="store_true", help="Run server in mock mode (without loading weights)")

    args = parser.parse_args()

    print(f"🚀 Starting Scalable LLM API Server on {args.host}:{args.port}")
    print(f"📦 Model Path: {args.model}")
    print(f"⚙️ Mock Mode: {args.mock}")

    builder = APIBuilder(
        model_path=args.model,
        host=args.host,
        port=args.port,
        workers=args.workers,
        mock_mode=args.mock,
    )
    builder.run()


if __name__ == "__main__":
    main()
