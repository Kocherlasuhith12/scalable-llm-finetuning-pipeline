#!/usr/bin/env bash
# Launch distributed training (multi-GPU / multi-node).
# Usage: ./scripts/launch_distributed_training.sh [--config configs/lora_config.yaml] [--num-gpus N]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/lora_config.yaml}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_$(date +%Y%m%d_%H%M%S)}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "Launching distributed training with $NUM_GPUS GPUs..."
  torchrun --nproc_per_node="$NUM_GPUS" \
    -m workflows.dags.training_pipeline \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR"
else
  echo "Launching single-GPU training..."
  python -m workflows.dags.training_pipeline --config "$CONFIG" --output-dir "$OUTPUT_DIR"
fi

echo "Training complete. Output: $OUTPUT_DIR"
