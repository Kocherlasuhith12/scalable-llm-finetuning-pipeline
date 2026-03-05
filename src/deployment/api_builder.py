"""API endpoint builder for model serving."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class APIBuilder:
    """Build FastAPI/Starlette serving API for a trained model."""

    def __init__(
        self,
        model_path: str,
        framework: str = "fastapi",
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.framework = framework
        self.host = host
        self.port = port
        self.workers = workers

    def create_app(self) -> Any:
        """Create ASGI app with /generate and /health endpoints."""
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
            app = FastAPI(title="LLM Fine-tuning Pipeline API")
            model_ref: Any = None
            tokenizer_ref: Any = None

            @app.on_event("startup")
            async def load_model():
                nonlocal model_ref, tokenizer_ref
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer_ref = AutoTokenizer.from_pretrained(self.model_path)
                model_ref = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype="auto",
                )

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            @app.post("/generate")
            async def generate(request: dict):
                prompt = request.get("prompt", "")
                max_tokens = request.get("max_tokens", 256)
                if not model_ref or not tokenizer_ref:
                    return JSONResponse({"error": "Model not loaded"}, status_code=503)
                inputs = tokenizer_ref(prompt, return_tensors="pt").to(model_ref.device)
                outputs = model_ref.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=request.get("do_sample", True),
                    temperature=request.get("temperature", 0.7),
                )
                text = tokenizer_ref.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return {"generated_text": text}

            return app
        except ImportError as e:
            logger.warning("FastAPI not available: %s", e)
            return None

    def run(self) -> None:
        """Run the API server."""
        app = self.create_app()
        if app:
            import uvicorn
            uvicorn.run(app, host=self.host, port=self.port, workers=self.workers)
