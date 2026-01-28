#!/usr/bin/env python3
"""
Robust & reproducible export:
PyTorch Wav2Vec2 -> ONNX

Behavior:
 - Try legacy ONNX export with dynamic axes for INPUT only (recommended)
 - If that fails, fall back to a fixed-shape ONNX export (still usable)
 - Validate with ONNX checker + shape inference
 - Save small metadata JSON for downstream tooling
 - Log to results/terminal_logs via your repo logger and print to stdout
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path

import torch
import onnx
import numpy as np
from transformers import AutoModelForCTC

# use your repo logger (signature: get_logger(name=None, log_dir=None, level=logging.INFO))
from code.utils.logger import get_logger

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
logger = get_logger(name="export_to_onnx", level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ONNX_DIR = PROJECT_ROOT / "models" / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = ONNX_DIR / "wav2vec2.onnx"
METADATA_PATH = ONNX_DIR / "wav2vec2.onnx.meta.json"

MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
OPSET = 18
SEED = 42
DUMMY_SEQ_LEN = 16000  # 1s audio @16kHz

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Small helper to print+log (keeps verbose terminal output)
def info(msg: str):
    print(msg)
    logger.info(msg)

def warn(msg: str):
    print(msg)
    logger.warning(msg)

def err(msg: str):
    print(msg, file=sys.stderr)
    logger.error(msg)

# ---------------------------------------------------------------------
# Export strategies
# ---------------------------------------------------------------------
def export_with_dynamic_input_only(model, dummy_input, out_path: Path):
    """
    Preferred: dynamic axes only for input (avoid specifying outputs).
    This avoids PyTorch torch.export/dynamic_shapes validation errors.
    """
    info("Attempting ONNX export: dynamic axes (INPUT only)")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        # IMPORTANT: only input is dynamic (batch,time)
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
        },
        opset_version=OPSET,
        verbose=False,
    )

def export_fixed_shape(model, dummy_input, out_path: Path):
    """
    Fallback: fixed shape export (no dynamic_axes).
    This produces a deterministic ONNX usable for builds, but with
    fixed time-dimension (DUMMY_SEQ_LEN).
    """
    info("Attempting ONNX export: fixed-shape (no dynamic axes)")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        opset_version=OPSET,
        verbose=False,
    )

# ---------------------------------------------------------------------
# Validate & shape-infer
# ---------------------------------------------------------------------
def validate_and_infer(onnx_path: Path):
    info("Running ONNX checker & shape inference")
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        # shape inference
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            onnx.save(inferred, str(onnx_path))
            info("Shape inference applied and saved to ONNX file")
            model = inferred
        except Exception as e:
            warn(f"Shape inference failed or not applicable: {e} (continuing with original ONNX)")
        # re-run checker just in case
        onnx.checker.check_model(model)
        info("ONNX model validated successfully")
        return model
    except Exception as e:
        err(f"ONNX validation failed: {e}")
        raise

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    start_ts = time.time()
    info("=== Export PyTorch -> ONNX (robust) ===")
    info(f"Project root : {PROJECT_ROOT}")
    info(f"Model        : {MODEL_NAME}")
    info(f"Output ONNX  : {ONNX_PATH}")
    info(f"Opset        : {OPSET}")

    # Load model to CPU for deterministic export
    info("Loading PyTorch model (CPU, eval mode). This may download weights the first time.")
    try:
        model = AutoModelForCTC.from_pretrained(MODEL_NAME)
    except Exception as e:
        err(f"Failed to load model '{MODEL_NAME}': {e}")
        raise SystemExit(2)
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, DUMMY_SEQ_LEN, dtype=torch.float32)

    # Try preferred export strategy first
    success = False
    last_exception = None

    try:
        export_with_dynamic_input_only(model, dummy_input, ONNX_PATH)
        success = True
        info("Primary export (dynamic input) succeeded.")
    except Exception as e:
        last_exception = e
        warn("Primary export (dynamic input) failed. Error:")
        logger.exception(e)
        # Fall through to fallback attempt

    if not success:
        try:
            export_fixed_shape(model, dummy_input, ONNX_PATH)
            success = True
            info("Fallback export (fixed-shape) succeeded.")
            warn("Note: ONNX has fixed sequence length (time dim). You may need to adapt TensorRT shapes or rebuild ONNX for other lengths.")
        except Exception as e:
            last_exception = e
            err("Fallback export also failed. See traceback below.")
            logger.exception(e)

    if not success:
        err("Both export attempts failed. Aborting.")
        raise SystemExit(3)

    # Validate ONNX
    try:
        onnx_model = validate_and_infer(ONNX_PATH)
    except Exception:
        err("Final ONNX is invalid. Aborting.")
        raise SystemExit(4)

    # Save metadata for downstream steps (useful for pipeline)
    try:
        meta = {
            "onnx_path": str(ONNX_PATH),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "opset": OPSET,
            "inputs": [i.name for i in onnx_model.graph.input],
            "outputs": [o.name for o in onnx_model.graph.output],
            "initializer_count": len(onnx_model.graph.initializer),
        }
        with open(METADATA_PATH, "w") as fh:
            json.dump(meta, fh, indent=2)
        info(f"Saved ONNX metadata → {METADATA_PATH}")
    except Exception as e:
        warn(f"Failed to write metadata JSON: {e}")

    elapsed = time.time() - start_ts
    info(f"Export completed in {elapsed:.1f}s")
    info("=== Export finished ===")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc or 0)
    except SystemExit as se:
        raise
    except Exception as e:
        logger.exception("Unhandled exception during ONNX export")
        print("Unhandled exception:", e, file=sys.stderr)
        sys.exit(10)