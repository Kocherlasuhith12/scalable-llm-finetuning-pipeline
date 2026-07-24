# 06. Training Pipeline

## Fine-Tuning Methods
1. **Supervised Fine-Tuning (SFT)**: Full fine-tuning of model parameters using PyTorch and Hugging Face `TRL` `SFTTrainer`.
2. **LoRA (Low-Rank Adaptation)**: Injects rank-decomposition matrices into target linear modules (e.g. `q_proj`, `v_proj`, `k_proj`, `o_proj`), dramatically reducing trainable parameters.
3. **QLoRA (Quantized LoRA)**: Loads base model in 4-bit NormalFloat (NF4) with Double Quantization and paged optimizers (`bitsandbytes`), enabling fine-tuning on limited GPU VRAM.
4. **Direct Preference Optimization (DPO)**: Optimizes model alignment directly on preference pairs (`chosen` vs `rejected`) without needing a separate reward model.

## Training Workflow & Callbacks
- **Step Telemetry**: Custom `MetricsLoggerCallback` emits loss, learning rate, epoch, step, and GPU memory usage at configurable logging steps.
- **Checkpointing**: Saves top-K checkpoints evaluated against validation loss.
- **Early Stopping**: Halts training when validation loss fails to improve across consecutive evaluation windows.
