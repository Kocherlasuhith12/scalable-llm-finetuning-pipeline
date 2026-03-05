#!/usr/bin/env python3
"""Merge LoRA/QLoRA adapters into base model and save full model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Merge PEFT adapters into base model")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter-path", type=str, required=True, help="Path to saved adapter (e.g. checkpoint or PEFT output)")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save merged model")
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("Merging...")
    model = model.merge_and_unload()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(out)
    print(f"Merged model saved to {out}")


if __name__ == "__main__":
    main()
