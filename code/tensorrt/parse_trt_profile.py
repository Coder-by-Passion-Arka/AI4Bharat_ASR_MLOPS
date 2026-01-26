#!/usr/bin/env python3
"""
Parse TensorRT profile JSON and save top-k operators to CSV.
Usage:
    python parse_trt_profile.py <profile.json> <output.csv> [--top K]
"""
import json
import csv
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python parse_trt_profile.py <profile.json> <output.csv> [--top K]")
    sys.exit(1)

profile_path = sys.argv[1]
output_csv = sys.argv[2]
top_k = 30
if len(sys.argv) == 4:
    try:
        top_k = int(sys.argv[3])
    except Exception:
        pass

if not os.path.exists(profile_path):
    print(f"[TensorRT] ERROR: profile not found: {profile_path}")
    sys.exit(2)

with open(profile_path, "r") as f:
    try:
        data = json.load(f)
    except Exception as e:
        print(f"[TensorRT] ERROR: failed to read JSON: {e}")
        sys.exit(3)

# data is typically a list of layer dicts; tolerate other shapes
if not isinstance(data, list):
    # try to find nested lists
    found = None
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                found = v
                break
    if found is None:
        print("[TensorRT] ERROR: unexpected profile structure.")
        sys.exit(4)
    data = found

# sort by timeMs (fallback to 'time' or 'time_ms')
def get_time(entry):
    for k in ("timeMs", "time_ms", "time"):
        if k in entry:
            return float(entry[k])
    return 0.0

top_layers = sorted(data, key=get_time, reverse=True)[:top_k]

os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer_name", "time_ms"])
    for layer in top_layers:
        name = layer.get("name", layer.get("op", "unknown"))
        writer.writerow([name, get_time(layer)])

print(f"[TensorRT] Parsed profile -> {output_csv}")
