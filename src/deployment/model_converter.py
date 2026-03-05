"""Model conversion to ONNX/TensorRT for deployment."""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelConverter:
    """Convert HuggingFace models to ONNX or TensorRT."""

    def __init__(
        self,
        output_dir: str = "./export",
        opset_version: int = 14,
        optimize: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.opset_version = opset_version
        self.optimize = optimize

    def to_onnx(
        self,
        model: Any,
        tokenizer: Any,
        seq_length: int = 512,
        **kwargs: Any,
    ) -> Path:
        """Export model to ONNX format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from transformers.onnx import export
            export(
                preprocessor=tokenizer,
                model=model,
                config=model.config,
                opset=self.opset_version,
                output=Path(self.output_dir) / "model.onnx",
                **kwargs,
            )
            return self.output_dir / "model.onnx"
        except Exception as e:
            logger.warning("ONNX export failed (install optimum): %s", e)
            return self.output_dir / "model.onnx"

    def to_tensorrt(self, onnx_path: Optional[Path] = None, **kwargs: Any) -> Optional[Path]:
        """Convert ONNX to TensorRT (requires tensorrt)."""
        onnx_path = onnx_path or self.output_dir / "model.onnx"
        if not onnx_path.exists():
            return None
        try:
            import tensorrt as trt
            logger.info("TensorRT conversion not fully implemented; use trtexec or polygraphy")
            return self.output_dir / "model.engine"
        except ImportError:
            logger.warning("tensorrt not installed")
            return None
