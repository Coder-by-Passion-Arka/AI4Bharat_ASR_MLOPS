#!/usr/bin/env python3
"""
ONNX optimization stage.

Purpose:
- Validate exported ONNX model
- Optionally run lightweight graph cleanup
- Produce a stable ONNX artifact for TensorRT builds

NOTE:
- INT8 quantization is intentionally NOT performed.
- FP16 / Mixed Precision are handled at TensorRT engine build time.
"""

import os
import shutil
import onnx

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

ONNX_DIR = os.path.join(PROJECT_ROOT, "models", "onnx")
SRC = os.path.join(ONNX_DIR, "wav2vec2.onnx")
DST = os.path.join(ONNX_DIR, "wav2vec2_optimized.onnx")

print(f"[onnx_optim] Source ONNX: {SRC}")

if not os.path.isfile(SRC):
    raise FileNotFoundError(f"[onnx_optim] ONNX model not found: {SRC}")

# ------------------------------------------------------------------
# Basic ONNX validation
# ------------------------------------------------------------------
try:
    model = onnx.load(SRC)
    onnx.checker.check_model(model)
    print("[onnx_optim] ONNX model passed checker.")
except Exception as e:
    raise RuntimeError(f"[onnx_optim] Invalid ONNX model: {e}")

# ------------------------------------------------------------------
# Optimization step (currently identity copy)
# ------------------------------------------------------------------
# Placeholder for future:
# - onnx-simplifier
# - constant folding
# - shape inference

shutil.copyfile(SRC, DST)

print(f"[onnx_optim] Optimized ONNX written to: {DST}")
print("[onnx_optim] NOTE: Precision optimization handled by TensorRT (FP16 / Mixed).")