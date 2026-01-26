#!/usr/bin/env python3
"""
Compare end-to-end latency across backends:
PyTorch, ONNX Runtime, TensorRT FP16, TensorRT Mixed
"""

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

data = [
    {"Backend": "PyTorch", "Latency_ms": float(open(os.path.join(RESULTS_DIR, "latency_pytorch.txt")).read())},
    {"Backend": "ONNX Runtime", "Latency_ms": float(open(os.path.join(RESULTS_DIR, "latency_onnx.txt")).read())},
    {"Backend": "TensorRT FP16", "Latency_ms": float(open(os.path.join(RESULTS_DIR, "latency_trt_fp16.txt")).read())},
    {"Backend": "TensorRT Mixed", "Latency_ms": float(open(os.path.join(RESULTS_DIR, "latency_trt_mixed.txt")).read())},
]

df = pd.DataFrame(data)
baseline = df[df["Backend"] == "PyTorch"]["Latency_ms"].values[0]
df["Speedup_vs_PyTorch"] = baseline / df["Latency_ms"]

out_csv = os.path.join(RESULTS_DIR, "backend_latency_comparison.csv")
df.to_csv(out_csv, index=False)

print("\nBackend Latency Comparison:\n")
print(df)
print(f"\n[compare_backends] Saved to {out_csv}")
