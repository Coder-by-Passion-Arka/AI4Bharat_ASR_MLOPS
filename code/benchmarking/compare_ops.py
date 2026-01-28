#!/usr/bin/env python3
"""
Compare TensorRT FP16 vs Mixed Precision operator timings.

Outputs:
- Console table
- CSV comparison
- Bar chart (.png) for top-k most expensive layers
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from code.utils.logger import get_logger

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = get_logger(
    name="compare_ops",
    log_file="compare_ops.log"
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FP16_CSV = os.path.join(RESULTS_DIR, "trt_fp16_ops.csv")
MIXED_CSV = os.path.join(RESULTS_DIR, "trt_mixed_ops.csv")

# -----------------------------------------------------------------------------
# Load CSVs safely
# -----------------------------------------------------------------------------
def load_ops_csv(path, label):
    if not os.path.exists(path):
        logger.error(f"{label} CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {label} ops CSV ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to read {label} CSV: {e}")
        return None

fp16 = load_ops_csv(FP16_CSV, "FP16")
mixed = load_ops_csv(MIXED_CSV, "Mixed")

if fp16 is None or mixed is None:
    logger.error("Required TensorRT operator CSVs missing. Aborting comparison.")
    raise SystemExit(1)

# -----------------------------------------------------------------------------
# Validate schema
# -----------------------------------------------------------------------------
required_cols = {"layer_name", "time_ms"}
for name, df in [("FP16", fp16), ("Mixed", mixed)]:
    if not required_cols.issubset(df.columns):
        logger.error(f"{name} CSV missing required columns: {required_cols}")
        raise SystemExit(2)

# -----------------------------------------------------------------------------
# Merge & compute speedup
# -----------------------------------------------------------------------------
logger.info("Merging FP16 and Mixed operator timings")

merged = fp16.merge(
    mixed,
    on="layer_name",
    suffixes=("_fp16", "_mixed")
)

# avoid division by zero
merged = merged[merged["time_ms_fp16"] > 0]

merged["speedup_fp16_vs_mixed"] = (
    merged["time_ms_mixed"] / merged["time_ms_fp16"]
)

# -----------------------------------------------------------------------------
# Select top-K heavy ops (by FP16 time)
# -----------------------------------------------------------------------------
TOP_K = 10
merged = merged.sort_values("time_ms_fp16", ascending=False).head(TOP_K)

# -----------------------------------------------------------------------------
# Save merged CSV
# -----------------------------------------------------------------------------
out_csv = os.path.join(RESULTS_DIR, "trt_fp16_vs_mixed_ops.csv")
merged.to_csv(out_csv, index=False)
logger.info(f"Saved merged ops comparison CSV → {out_csv}")

# -----------------------------------------------------------------------------
# Console output
# -----------------------------------------------------------------------------
logger.info("Top TensorRT Layer Comparison (FP16 vs Mixed):")
logger.info("\n" + merged.to_string(index=False))

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 6))

ax = merged.set_index("layer_name")[[
    "time_ms_fp16", "time_ms_mixed"
]].plot.bar(ax=plt.gca())

plt.ylabel("Execution Time (ms)")
plt.title(
    "TensorRT Operator-Level Comparison\n"
    "FP16 vs Mixed Precision (Top Heavy Layers)"
)
plt.xticks(rotation=30, ha="right")

# annotate bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", fontsize=8)

plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "trt_fp16_vs_mixed.png")
plt.savefig(plot_path, dpi=150)
plt.close()

logger.info(f"Saved operator comparison plot → {plot_path}")
logger.info("========== TensorRT operator comparison completed ==========")