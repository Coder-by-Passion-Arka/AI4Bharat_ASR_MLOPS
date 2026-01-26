#!/usr/bin/env python3
"""
PyTorch profiling for IndicWav2Vec ASR model.

This script helps identify:
- Slow layers
- GPU vs CPU usage
- Memory bottlenecks

Run this BEFORE any optimization.

Outputs:
- results/pytorch_profiler.txt  (human-readable table)
- results/pytorch_ops.csv       (machine-readable)
"""

import os
import csv
import torch
from transformers import AutoModelForCTC, AutoProcessor

# -----------------------------
# Configuration
# -----------------------------
print("\t\tSystem Configuration")
print("=" * 60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script is present in: {SCRIPT_DIR}")

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
print(f"Root of the Project is: {PROJECT_ROOT}")

PT_DIR = os.path.join(PROJECT_ROOT, "models", "pytorch")
os.makedirs(PT_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved in: {RESULTS_DIR}")

MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print("=" * 60)

# -----------------------------
# Load model and processor
# -----------------------------
print(f"[profiler] Device: {DEVICE}. Model: {MODEL_NAME}")

model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

# -----------------------------
# Dummy audio input (shape only)
# -----------------------------
# 1 second of audio @ 16kHz
dummy_audio = torch.randn(1, 16000, device=DEVICE)

# -----------------------------
# Warm-up (important for GPU)
# -----------------------------
with torch.no_grad():
    for _ in range(3):
        _ = model(dummy_audio).logits

# -----------------------------
# PyTorch Profiler
# -----------------------------
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
        if DEVICE == "cuda"
        else torch.profiler.ProfilerActivity.CPU,
    ],
    record_shapes=True,
    profile_memory=True,
) as profiler:
    with torch.no_grad():
        _ = model(dummy_audio).logits

# -----------------------------
# Save the Profiling Table (With Top 30 Operations)
# -----------------------------
# -----------------------------
# Collect profiler statistics
# -----------------------------
def get_event_time(evt):
    """
    Return best available time metric for a profiler event.
    Works across PyTorch versions.
    """
    if torch.cuda.is_available():
        if hasattr(evt, "cuda_time_total"):
            return evt.cuda_time_total
        if hasattr(evt, "cuda_time"):
            return evt.cuda_time
        if hasattr(evt, "self_cuda_time"):
            return evt.self_cuda_time
    return evt.cpu_time_total


events = profiler.key_averages()

# Convert to list and sort safely
sorted_events = sorted(
    events,
    key=lambda e: get_event_time(e),
    reverse=True
)

top_k = 30
top_events = sorted_events[:top_k]

# 1) Terminal Table (TXT)
table = profiler.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=30
)

out_txt = os.path.join(RESULTS_DIR, "pytorch_profiler.txt")
with open(out_txt, "w") as f:
    f.write(table)

print(f"[profiler] Written profiler table to: {out_txt}\n")
print(table)

# 2) Save machine-readable CSV
csv_path = os.path.join(RESULTS_DIR, "pytorch_ops.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "op_name",
        "cpu_time_total_ms",
        "self_cpu_time_ms",
        "device_time_ms",
        "self_device_time_ms",
        "calls"
    ])

    for evt in sorted_events:
        writer.writerow([
            evt.key,
            evt.cpu_time_total / 1e3, # micro seconds to mili seconds
            evt.self_cpu_time_total / 1e3,
            evt.device_time / 1e3, 
            evt.device_time / 1e3,
            evt.count
        ])

print(f"[profiler] Operator CSV saved to: {csv_path}")

# 3) Compact top CUDA consumers (terminal)
if DEVICE == "cuda":
    print("\n[profiler] Top CUDA consumers (ms):")
    for evt in top_events[:8]:
        print(
            f"  {evt.key:30s} "
            f"{get_event_time(evt)/1e3:8.3f} ms "
            f"(calls: {evt.count})"
        )

# -----------------------------
# Save the PyTorch model
# -----------------------------
# traced = torch.jit.trace(model, dummy_audio)
# pt_path = os.path.join(PT_DIR, "wav2vec2.pt")
# traced.save(pt_path)

# print(f"[PyTorch] TorchScript model saved at {pt_path}")

pt_path = os.path.join(PT_DIR, "wav2vec2_state_dict.pt")
torch.save(model.state_dict(), pt_path)

print(f"[PyTorch] state_dict saved at {pt_path}")