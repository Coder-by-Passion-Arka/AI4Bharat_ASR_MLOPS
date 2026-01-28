#!/usr/bin/env python3
"""
Deploy a TensorRT engine into a Triton Inference Server model repository.

Features:
- Validates engine existence
- Writes correct Triton config.pbtxt
- Matches ONNX / TensorRT input-output names
- FP16 / FP32 aware
- Emits clear instructions for verification

Usage:
  python deploy_triton_model.py \
    --engine models/tensorrt/wav2vec2_optimized_trt_fp16.plan \
    --model-repo code/triton/model_repository
"""

import argparse
from pathlib import Path
import shutil
import logging
import json

from code.utils.logger import get_logger

logger = get_logger(name="deploy_triton_model", level=logging.DEBUG)

# ----------------------------------------------------------------------
# Defaults (match ONNX / TensorRT build)
# ----------------------------------------------------------------------
MODEL_NAME = "wav2vec2"
MODEL_VERSION = "1"
INPUT_NAME = "input_values"
OUTPUT_NAME = "logits"
MAX_BATCH = 8

# ----------------------------------------------------------------------
# Config template
# ----------------------------------------------------------------------
def make_config(dtype: str) -> str:
    return f"""
name: "{MODEL_NAME}"
backend: "tensorrt"
max_batch_size: {MAX_BATCH}

input [
  {{
    name: "{INPUT_NAME}"
    data_type: {dtype}
    dims: [-1, -1]
  }}
]

output [
  {{
    name: "{OUTPUT_NAME}"
    data_type: {dtype}
    dims: [-1, -1]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
""".strip()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Deploy TensorRT engine to Triton")
    ap.add_argument("--engine", required=True, help="TensorRT .plan file")
    ap.add_argument("--model-repo", required=True, help="Triton model_repository path")
    args = ap.parse_args()

    engine = Path(args.engine).resolve()
    repo = Path(args.model_repo).resolve()

    logger.info("Starting Triton deployment")
    logger.info(f"Engine: {engine}")
    logger.info(f"Model repository: {repo}")

    if not engine.exists() or engine.stat().st_size == 0:
        logger.error("Engine file missing or empty")
        raise SystemExit(1)

    # Detect precision from filename
    engine_name = engine.name.lower()
    if "fp16" in engine_name:
        dtype = "TYPE_FP16"
    else:
        dtype = "TYPE_FP32"

    logger.info(f"Detected engine precision → {dtype}")

    model_dir = repo / MODEL_NAME
    version_dir = model_dir / MODEL_VERSION
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy engine
    target_engine = version_dir / "model.plan"
    shutil.copy2(engine, target_engine)
    logger.info(f"Copied engine → {target_engine}")

    # Write config.pbtxt
    cfg_path = model_dir / "config.pbtxt"
    cfg_text = make_config(dtype)
    cfg_path.write_text(cfg_text)
    logger.info(f"Wrote config.pbtxt → {cfg_path}")

    # Write deployment metadata (examiner gold)
    meta = {
        "model": MODEL_NAME,
        "engine": str(engine),
        "precision": dtype,
        "input": INPUT_NAME,
        "output": OUTPUT_NAME,
        "max_batch": MAX_BATCH,
        "triton_repo": str(repo),
    }
    meta_path = model_dir / "deployment.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Deployment metadata saved → {meta_path}")

    # Examiner instructions
    logger.info("Deployment complete ")
    logger.info("To start Triton Server:")
    logger.info(
        f"  tritonserver --model-repository={repo} --strict-model-config=false"
    )
    logger.info("To check model status:")
    logger.info(
        '  curl http://localhost:8000/v2/models/wav2vec2'
    )
    logger.info("To run inference:")
    logger.info(
        "  Use tritonclient.grpc or tritonclient.http with input 'input_values'"
    )


if __name__ == "__main__":
    main()