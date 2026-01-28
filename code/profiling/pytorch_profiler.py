#!/usr/bin/env python3
"""
PyTorch profiling for IndicWav2Vec ASR model.

Outputs:
- results/pytorch_profiler.txt   (human-readable table)
- results/pytorch_ops.csv        (operator-level timings)
- results/latency_pytorch.txt    (end-to-end latency, seconds)
"""

import os
import csv
import time
import torch
from transformers import AutoModelForCTC, AutoProcessor

from code.utils.logger import get_logger

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = get_logger(
    name="pytorch_profiler",
    log_file="pytorch_profiler.log"
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

PT_DIR = os.path.join(PROJECT_ROOT, "models", "pytorch")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(PT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_event_time_ms(evt) -> float:
    """
    Robustly extract the best available time metric (ms) across PyTorch versions.
    Prefers CUDA time when available.
    """
    if torch.cuda.is_available():
        for attr in ("cuda_time_total", "self_cuda_time", "cuda_time"):
            if hasattr(evt, attr):
                return getattr(evt, attr) / 1e3
    return evt.cpu_time_total / 1e3

# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------
logger.info("=" * 70)
logger.info("PyTorch Profiling – IndicWav2Vec ASR")
logger.info("=" * 70)
logger.info(f"Project root : {PROJECT_ROOT}")
logger.info(f"Model        : {MODEL_NAME}")
logger.info(f"Device       : {DEVICE}")

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
logger.info("Loading model and processor")

model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

# -----------------------------------------------------------------------------
# Dummy input (1 second @ 16kHz)
# -----------------------------------------------------------------------------
dummy_audio = torch.randn(1, 16000, device=DEVICE)

# -----------------------------------------------------------------------------
# Warm-up (important for GPU stability)
# -----------------------------------------------------------------------------
logger.info("Running warm-up iterations")
with torch.no_grad():
    for _ in range(3):
        _ = model(dummy_audio).logits

# -----------------------------------------------------------------------------
# Measure end-to-end latency
# -----------------------------------------------------------------------------
logger.info("Measuring end-to-end PyTorch inference latency")

runs = 20
with torch.no_grad():
    start = time.time()
    for _ in range(runs):
        _ = model(dummy_audio).logits
    end = time.time()

avg_latency = (end - start) / runs
lat_path = os.path.join(RESULTS_DIR, "latency_pytorch.txt")
with open(lat_path, "w") as f:
    f.write(f"{avg_latency:.6f}")

logger.info(f"Average PyTorch latency: {avg_latency:.6f} s")
logger.info(f"Saved latency → {lat_path}")

# -----------------------------------------------------------------------------
# PyTorch Profiler
# -----------------------------------------------------------------------------
logger.info("Starting PyTorch profiler")

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

events = profiler.key_averages()

# -----------------------------------------------------------------------------
# Sort events by dominant time
# -----------------------------------------------------------------------------
sorted_events = sorted(
    events,
    key=get_event_time_ms,
    reverse=True
)

top_k = 30
top_events = sorted_events[:top_k]

# -----------------------------------------------------------------------------
# Save human-readable table
# -----------------------------------------------------------------------------
table = profiler.key_averages().table(
    sort_by="self_cpu_time_total" if DEVICE != "cuda" else "self_cuda_time_total",
    row_limit=top_k
)

txt_path = os.path.join(RESULTS_DIR, "pytorch_profiler.txt")
# with open(txt_path, "w") as f:
#     f.write(table)

# print(f"\nModel Profiling Stats: \n{table}")
logger.info("\nModel Profiling Stats:" + table)

logger.info(f"Saved profiler table → {txt_path}")

# -----------------------------------------------------------------------------
# Save CSV (machine-readable)
# -----------------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "pytorch_ops.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "op_name",
        "time_ms",
        "cpu_time_total_ms",
        "cuda_time_total_ms",
        "calls"
    ])

    for evt in sorted_events:
        writer.writerow([
            evt.key,
            get_event_time_ms(evt),
            evt.cpu_time_total / 1e3,
            getattr(evt, "cuda_time_total", 0.0) / 1e3,
            evt.count
        ])

logger.info(f"Saved operator CSV → {csv_path}")

# -----------------------------------------------------------------------------
# Top CUDA ops (console-friendly)
# -----------------------------------------------------------------------------
if DEVICE == "cuda":
    logger.info("Top CUDA-consuming operators:")
    for evt in top_events[:8]:
        logger.info(
            f"{evt.key:40s} "
            f"{get_event_time_ms(evt):8.3f} ms "
            f"(calls: {evt.count})"
        )

# -----------------------------------------------------------------------------
# Save model state_dict
# -----------------------------------------------------------------------------
pt_path = os.path.join(PT_DIR, "wav2vec2_state_dict.pt")
torch.save(model.state_dict(), pt_path)
logger.info(f"Saved PyTorch state_dict → {pt_path}")

logger.info("PyTorch profiling completed successfully")
logger.info("=" * 70)