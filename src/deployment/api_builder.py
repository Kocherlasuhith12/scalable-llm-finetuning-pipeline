"""API endpoint builder for model serving with OpenAI compatibility."""

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class APIBuilder:
    """Build FastAPI serving API for a trained model with OpenAI compatibility."""

    def __init__(
        self,
        model_path: str,
        framework: str = "fastapi",
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        mock_mode: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.framework = framework
        self.host = host
        self.port = port
        self.workers = workers
        self.mock_mode = mock_mode

    def create_app(self) -> Any:
        """Create ASGI app with /health, /generate, /v1/models and /v1/chat/completions endpoints."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse

            app = FastAPI(
                title="Scalable LLM Fine-tuning Pipeline API",
                description="Production API server for fine-tuned LLM inference",
                version="1.0.0",
            )

            # Enable CORS for live web apps and clients
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            state: Dict[str, Any] = {
                "model": None,
                "tokenizer": None,
                "loaded": False,
                "model_name": str(self.model_path.name if self.model_path else "llm-fine-tuned"),
            }

            @app.on_event("startup")
            async def load_model():
                if self.mock_mode:
                    logger.info("Running in Mock Mode. Skipping heavy model weight loading.")
                    state["loaded"] = True
                    return

                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    logger.info("Loading tokenizer from %s...", self.model_path)
                    state["tokenizer"] = AutoTokenizer.from_pretrained(self.model_path)
                    logger.info("Loading model from %s...", self.model_path)
                    state["model"] = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        torch_dtype="auto",
                        low_cpu_mem_usage=True,
                    )
                    state["loaded"] = True
                    logger.info("Model successfully loaded for API serving.")
                except Exception as e:
                    logger.warning("Could not load real model weights (%s). Falling back to mock generator.", e)
                    state["loaded"] = True

            @app.get("/health")
            async def health():
                return {
                    "status": "ok",
                    "model_loaded": state["loaded"],
                    "model_name": state["model_name"],
                    "mock_mode": self.mock_mode or (state["model"] is None),
                }

            @app.get("/v1/models")
            async def list_models():
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": state["model_name"],
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "scalable-llm-pipeline",
                        }
                    ],
                }

            @app.post("/generate")
            async def generate(request: dict):
                prompt = request.get("prompt", "")
                max_tokens = request.get("max_tokens", 256)
                temperature = request.get("temperature", 0.7)

                if state["model"] is not None and state["tokenizer"] is not None:
                    inputs = state["tokenizer"](prompt, return_tensors="pt").to(state["model"].device)
                    outputs = state["model"].generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=request.get("do_sample", True),
                        temperature=temperature,
                    )
                    text = state["tokenizer"].decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                else:
                    text = f"[Mock Output for prompt: '{prompt[:50]}...'] Fine-tuned LLM response generated successfully."

                return {"generated_text": text, "prompt": prompt}

            @app.post("/v1/chat/completions")
            async def chat_completions(request: dict):
                messages: List[dict] = request.get("messages", [])
                max_tokens = request.get("max_tokens", 256)
                temperature = request.get("temperature", 0.7)

                # Format chat prompt from messages
                full_prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])

                if state["model"] is not None and state["tokenizer"] is not None:
                    inputs = state["tokenizer"](full_prompt, return_tensors="pt").to(state["model"].device)
                    outputs = state["model"].generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                    )
                    response_text = state["tokenizer"].decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    prompt_tokens = int(inputs["input_ids"].shape[1])
                    completion_tokens = int(outputs[0].shape[0] - prompt_tokens)
                else:
                    last_user_msg = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "Hello")
                    response_text = f"This is an automated response from your fine-tuned model for: '{last_user_msg}'"
                    prompt_tokens = len(full_prompt.split())
                    completion_tokens = len(response_text.split())

                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", state["model_name"]),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }

            return app
        except ImportError as e:
            logger.warning("FastAPI/Uvicorn not available: %s", e)
            return None

    def run(self) -> None:
        """Run the API server."""
        app = self.create_app()
        if app:
            import uvicorn
            uvicorn.run(app, host=self.host, port=self.port, workers=self.workers)

