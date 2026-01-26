#!/usr/bin/env python3
"""
Benchmark ONNX Runtime inference and produce
a per-operator latency breakdown (Top-30 ops).

Outputs a versus-ready table for analysis.
"""

import os
import csv
import time
import json
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from tabulate import tabulate

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "onnx", "wav2vec2.onnx")
PROFILE_PATH = os.path.join(PROJECT_ROOT, "results", "onnx_profile.json")
os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)

# -----------------------------
# Session with profiling ON
# -----------------------------
so = ort.SessionOptions()
so.enable_profiling = True

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# -----------------------------
# Dummy input
# -----------------------------
dummy_input = np.random.randn(1, 16000).astype(np.float32)

# -----------------------------
# Warm-up
# -----------------------------
for _ in range(5):
    session.run(None, {"input_values": dummy_input})

# -----------------------------
# Timed runs
# -----------------------------
runs = 20
start = time.time()
for _ in range(runs):
    session.run(None, {"input_values": dummy_input})
end = time.time()

avg_latency = (end - start) / runs
print(f"\n[ONNX Runtime] Average latency: {avg_latency:.4f} seconds")

# -----------------------------
# Collect profiling data
# -----------------------------
profile_file = session.end_profiling()

# Move profile to known location
os.replace(profile_file, PROFILE_PATH)

print(f"[ONNX Runtime] Profiling trace saved at: {PROFILE_PATH}")

# -----------------------------
# Parse profile JSON
# -----------------------------
with open(PROFILE_PATH, "r") as f:
    trace = json.load(f)

op_time = defaultdict(float)

for event in trace:
    if event.get("cat") == "Node":
        op_name = event.get("name", "UNKNOWN")
        duration_us = event.get("dur", 0)
        op_time[op_name] += duration_us

# Convert to ms
op_time_ms = {k: v / 1000.0 for k, v in op_time.items()}

# -----------------------------
# Top 30 ops
# -----------------------------
top_ops = sorted(
    op_time_ms.items(),
    key=lambda x: x[1],
    reverse=True
)[:30]

# -----------------------------
# Print table
# -----------------------------
table = [
    (op, f"{time_ms:.3f}")
    for op, time_ms in top_ops
]

print("\nTop 30 ONNX Runtime Operations by Time (ms):\n")
print(tabulate(table, headers=["Operation", "Total Time (ms)"], tablefmt="github"))

# Store the Table as .CSV for later analysis
csv_path = os.path.join(PROJECT_ROOT, "results", "onnx_ops.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["op_name", "total_time_ms"])
    for op, time_ms in top_ops:
        writer.writerow([op, time_ms])

print(f"[ONNX Runtime] Operator profile saved to {csv_path}")
