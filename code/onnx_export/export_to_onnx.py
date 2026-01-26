#!/usr/bin/env python3
"""
Export PyTorch Wav2Vec2 -> ONNX (robust & reproducible).
Saves to models/onnx/wav2vec2.onnx
"""

import os
import torch
import onnx
from transformers import AutoModelForCTC, AutoProcessor
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ONNX_DIR = os.path.join(PROJECT_ROOT, "models", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
ONNX_PATH = os.path.join(ONNX_DIR, "wav2vec2.onnx")

print(f"[onnx_export] Exporting {MODEL_NAME} to {ONNX_PATH}")

# Load model on CPU for deterministic export
model = AutoModelForCTC.from_pretrained(MODEL_NAME).to("cpu")
# processor = AutoProcessor.from_pretrained(MODEL_NAME)
# model.eval()

# Dummy input (shape contract only)
dummy_input = torch.randn(1, 16000, dtype=torch.float32)

OPSET = 18

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "time"},
    },
    opset_version=OPSET,
)

print(f"[onnx_export] ONNX model saved at {ONNX_PATH}")

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("[onnx_export] ONNX model passed basic checker.")
