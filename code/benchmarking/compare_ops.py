#!/usr/bin/env python3
"""
Compare TensorRT FP16 vs Mixed Precision operator timings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

fp16 = pd.read_csv(os.path.join(RESULTS_DIR, "trt_fp16_ops.csv"))
mixed = pd.read_csv(os.path.join(RESULTS_DIR, "trt_mixed_ops.csv"))

merged = fp16.merge(
    mixed, on="layer_name", suffixes=("_fp16", "_mixed")
)

merged["speedup_fp16_vs_mixed"] = (
    merged["time_ms_mixed"] / merged["time_ms_fp16"]
)

merged = merged.sort_values("time_ms_fp16", ascending=False).head(15)

print("\nTop TensorRT Layer Comparison (FP16 vs Mixed):\n")
print(merged)

# Plot
merged.set_index("layer_name")[[
    "time_ms_fp16", "time_ms_mixed"
]].plot.bar(figsize=(12, 6))

plt.ylabel("Time (ms)")
plt.title("TensorRT FP16 vs Mixed Precision (Top Layers)")
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "trt_fp16_vs_mixed.png")
plt.savefig(plot_path)

print(f"\n[compare_ops] Plot saved to {plot_path}")