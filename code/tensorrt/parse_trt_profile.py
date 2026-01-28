#!/usr/bin/env python3
"""
Parse TensorRT profile JSON:
- Aggregate per-layer execution time
- Save top-K operators to CSV
- Plot top-10 operators by time

Usage:
    python parse_trt_profile.py <profile.json> <output.csv> [top_k]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

from code.utils.logger import get_logger

logger = get_logger(name="parse_trt_profile", level=10)  # DEBUG

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
if len(sys.argv) < 3:
    logger.error("Usage: python parse_trt_profile.py <profile.json> <output.csv> [top_k]")
    sys.exit(1)

profile_path = Path(sys.argv[1]).resolve()
output_csv = Path(sys.argv[2]).resolve()
top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 30

logger.info(f"TensorRT profile  : {profile_path}")
logger.info(f"Output CSV       : {output_csv}")
logger.info(f"Top-K operators  : {top_k}")

if not profile_path.exists():
    logger.error(f"Profile JSON not found: {profile_path}")
    sys.exit(2)

# ---------------------------------------------------------------------
# Load JSON
# ---------------------------------------------------------------------
try:
    raw = json.loads(profile_path.read_text())
except Exception:
    logger.exception("Failed to read TensorRT profile JSON")
    sys.exit(3)

# ---------------------------------------------------------------------
# Normalize TensorRT profile format
# ---------------------------------------------------------------------
def extract_layers(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError("Unsupported TensorRT profile format")

try:
    layers_raw = extract_layers(raw)
except Exception:
    logger.exception("Unable to extract layers from TensorRT profile")
    sys.exit(4)

logger.info(f"Found {len(layers_raw)} raw layer entries")

# ---------------------------------------------------------------------
# Aggregate timings
# ---------------------------------------------------------------------
def get_time_ms(entry):
    for k in ("timeMs", "time_ms", "time", "latencyMs"):
        if k in entry:
            return float(entry[k])
    return 0.0

agg = defaultdict(float)

for layer in layers_raw:
    name = (
        layer.get("name")
        or layer.get("layerName")
        or layer.get("op")
        or "unknown"
    )
    t = get_time_ms(layer)
    if t > 0:
        agg[name] += t

if not agg:
    logger.error("No valid timing data found in TensorRT profile")
    sys.exit(5)

df = (
    pd.DataFrame(
        [(k, v) for k, v in agg.items()],
        columns=["layer_name", "time_ms"],
    )
    .sort_values("time_ms", ascending=False)
)

# ---------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------
output_csv.parent.mkdir(parents=True, exist_ok=True)
df.head(top_k).to_csv(output_csv, index=False)
logger.info(f"CSV written → {output_csv}")

# ---------------------------------------------------------------------
# Plot Top-10
# ---------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

suffix = output_csv.stem.replace("trt_", "")
plot_path = results_dir / f"trt_top10_ops_{suffix}.png"

top10 = df.head(10).iloc[::-1]

plt.figure(figsize=(12, 6))
plt.barh(top10["layer_name"], top10["time_ms"])
plt.xlabel("Execution Time (ms)")
plt.title("Top-10 TensorRT Operators by Time")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

logger.info(f"Top-10 operator plot saved → {plot_path}")
logger.info("TensorRT profile parsing completed successfully")