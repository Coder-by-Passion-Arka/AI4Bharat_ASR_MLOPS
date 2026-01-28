# #!/usr/bin/env python3
# """
# Benchmark ONNX Runtime inference and produce:
# 1) Average latency
# 2) Per-operator latency breakdown (Top-K)
# 3) CSV + PNG artifacts for examiner analysis
# """

# import os
# import csv
# import time
# import json
# import numpy as np
# import onnxruntime as ort
# from collections import defaultdict
# import matplotlib.pyplot as plt

# from code.utils.logger import get_logger

# # -----------------------------------------------------------------------------
# # Logger
# # -----------------------------------------------------------------------------
# logger = get_logger(
#     name="benchmark_latency",
#     log_file="benchmark_latency.log"
# )

# # -----------------------------------------------------------------------------
# # Paths
# # -----------------------------------------------------------------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "onnx", "wav2vec2_optimized.onnx")
# PROFILE_PATH = os.path.join(RESULTS_DIR, "onnx_profile.json")
# CSV_PATH = os.path.join(RESULTS_DIR, "onnx_ops.csv")
# LATENCY_TXT = os.path.join(RESULTS_DIR, "latency_onnx.txt")
# PLOT_PATH = os.path.join(RESULTS_DIR, "onnx_top_ops.png")

# os.makedirs(RESULTS_DIR, exist_ok=True)

# # -----------------------------------------------------------------------------
# # Sanity checks
# # -----------------------------------------------------------------------------
# if not os.path.exists(MODEL_PATH):
#     logger.error(f"ONNX model not found: {MODEL_PATH}")
#     raise FileNotFoundError(MODEL_PATH)

# logger.info(f"Using ONNX model: {MODEL_PATH}")

# # -----------------------------------------------------------------------------
# # ONNX Runtime Session (with profiling)
# # -----------------------------------------------------------------------------
# so = ort.SessionOptions()
# so.enable_profiling = True

# providers = [
#     "CUDAExecutionProvider",
#     "CPUExecutionProvider",
# ]

# session = ort.InferenceSession(
#     MODEL_PATH,
#     sess_options=so,
#     providers=providers
# )

# active_providers = session.get_providers()
# logger.info(f"ONNX Runtime active providers: {active_providers}")

# if "CUDAExecutionProvider" not in active_providers:
#     logger.warning("CUDAExecutionProvider NOT active – inference is running on CPU!")

# # -----------------------------------------------------------------------------
# # Dummy input
# # -----------------------------------------------------------------------------
# input_name = session.get_inputs()[0].name
# dummy_input = np.random.randn(1, 16000).astype(np.float32)

# # -----------------------------------------------------------------------------
# # Warm-up
# # -----------------------------------------------------------------------------
# logger.info("Running warm-up iterations...")
# for _ in range(5):
#     session.run(None, {input_name: dummy_input})

# # -----------------------------------------------------------------------------
# # Timed runs
# # -----------------------------------------------------------------------------
# runs = 20
# logger.info(f"Running {runs} timed inference runs...")

# start = time.time()
# for _ in range(runs):
#     session.run(None, {input_name: dummy_input})
# end = time.time()

# avg_latency = (end - start) / runs
# logger.info(f"Average ONNX Runtime latency: {avg_latency:.6f} mili_seconds")

# with open(LATENCY_TXT, "w") as f:
#     f.write(f"{avg_latency * 1000:.3f}")

# logger.info(f"Saved latency → {LATENCY_TXT}")

# # -----------------------------------------------------------------------------
# # Collect profiling data
# # -----------------------------------------------------------------------------
# profile_file = session.end_profiling()
# os.replace(profile_file, PROFILE_PATH)

# logger.info(f"Profiling trace saved at: {PROFILE_PATH}")

# # -----------------------------------------------------------------------------
# # Parse profile JSON
# # -----------------------------------------------------------------------------
# with open(PROFILE_PATH, "r") as f:
#     trace = json.load(f)

# op_time_us = defaultdict(float)

# for event in trace:
#     if event.get("cat") == "Node":
#         name = event.get("name", "UNKNOWN")
#         duration = float(event.get("dur", 0.0))
#         op_time_us[name] += duration

# op_time_ms = {k: v / 1000.0 for k, v in op_time_us.items()}

# # -----------------------------------------------------------------------------
# # Top-K ops
# # -----------------------------------------------------------------------------
# TOP_K = 30
# top_ops = sorted(
#     op_time_ms.items(),
#     key=lambda x: x[1],
#     reverse=True
# )[:TOP_K]

# # -----------------------------------------------------------------------------
# # Save CSV
# # -----------------------------------------------------------------------------
# with open(CSV_PATH, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["op_name", "total_time_ms"])
#     for op, t in top_ops:
#         writer.writerow([op, f"{t:.3f}"])

# logger.info(f"Operator profile CSV saved → {CSV_PATH}")

# # -----------------------------------------------------------------------------
# # Plot Top Ops
# # -----------------------------------------------------------------------------
# ops, times = zip(*top_ops)

# plt.figure(figsize=(10, 6))
# plt.barh(ops[::-1], times[::-1])
# plt.xlabel("Total Time (ms)")
# plt.title("Top 30 ONNX Runtime Operators by Time")
# plt.tight_layout()
# plt.savefig(PLOT_PATH, dpi=150)
# plt.close()

# logger.info(f"Operator latency plot saved → {PLOT_PATH}")

# logger.info("ONNX Runtime benchmarking completed successfully.")

#!/usr/bin/env python3
"""
Unified benchmarking for backends:
 - PyTorch (loads model & measures end-to-end inference)
 - ONNX Runtime (loads ONNX and profiles; produces operator CSV + PNG)
 - TensorRT (benchmarks any .plan engines using trtexec when available)
 - Triton (calls existing triton_check.py if present)

Outputs (all in PROJECT/results):
 - latency_pytorch.txt   (mili_seconds, float)
 - latency_onnx.txt      (mili_seconds, float)
 - latency_trt_<prec>.txt (mili_seconds, float)  (if trtexec found and engine exists)
 - latency_triton.json   (if triton_check.py ran)
 - onnx_profile.json, onnx_ops.csv, onnx_top_ops.png
 - tensorRT operator CSVs/plots if profile JSONs exist (via parse_trt_profile.py)
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# local logger utility (your logger.py expects get_logger(name=..., level=...))
from code.utils.logger import get_logger

logger = get_logger(name="benchmark_latency", level=20)  # INFO by default

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
TRT_DIR = MODELS_DIR / "tensorrt"
os.makedirs(RESULTS_DIR, exist_ok=True)

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Unified backend latency benchmarking")
    p.add_argument("--runs", type=int, default=20, help="Number of timed runs")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up runs")
    p.add_argument("--force", action="store_true", help="Force re-run even if latency files exist")
    return p.parse_args()


# ---------------------------
# Helpers
# ---------------------------
def write_latency_file(path: Path, mili_seconds: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{mili_seconds:.6f}")
    logger.info(f"Wrote latency file: {path}  (mili_seconds)")

def file_exists_and_not_forced(path: Path, force: bool):
    return path.exists() and not force

def run_subprocess(cmd, cwd=None, capture_output=False, text=True, timeout=None):
    logger.debug(f"Running subprocess: {' '.join(map(str, cmd))} (cwd={cwd})")
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=text, timeout=timeout)
        return res
    except Exception as e:
        logger.warning(f"Subprocess failed: {e}")
        raise

# ---------------------------
# PyTorch benchmarking
# ---------------------------
def benchmark_pytorch(runs: int, warmup: int, force: bool):
    lat_file = RESULTS_DIR / "latency_pytorch.txt"
    if file_exists_and_not_forced(lat_file, force):
        logger.info(f"PyTorch latency already exists at {lat_file}. Skipping (use --force to re-run).")
        return

    try:
        import torch
        from transformers import AutoModelForCTC
    except Exception as e:
        logger.error(f"PyTorch benchmark skipped: missing dependency ({e})")
        return

    MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Benchmarking PyTorch (model={MODEL_NAME}) on device={device}")

    model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # dummy audio (1 x 16000)
    dummy = torch.randn(1, 16000, device=device)

    # warmup
    logger.info(f"PyTorch warmup: {warmup} runs")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy).logits

    # timed runs
    logger.info(f"PyTorch timed runs: {runs}")
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy).logits
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()

    avg_ms = (t1 - t0) / runs * 1000 # For milisecond conversion
    write_latency_file(lat_file, avg_ms)
    logger.info(f"PyTorch average latency: {avg_ms:.6f} ms")

# ---------------------------
# ONNX Runtime benchmarking + profile parsing
# ---------------------------
def benchmark_onnx(runs: int, warmup: int, force: bool):
    try:
        import onnxruntime as ort
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error(f"ONNX benchmark skipped: missing dependency ({e})")
        return

    # Prefer optimized ONNX if present
    model_path = ONNX_DIR / "wav2vec2_optimized.onnx"
    if not model_path.exists():
        model_path = ONNX_DIR / "wav2vec2.onnx"
    if not model_path.exists():
        raise FileNotFoundError("No ONNX model found in models/onnx")

    lat_file = RESULTS_DIR / "latency_onnx.txt"
    if lat_file.exists() and not force:
        logger.info(f"ONNX latency already exists at {lat_file} (use --force to re-run).")
        return

    logger.info(f"Using ONNX model: {model_path}")

    so = ort.SessionOptions()
    so.enable_profiling = True

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)

    active = session.get_providers()
    logger.info(f"ONNX Runtime active providers: {active}")

    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider is NOT available. "
            "You are benchmarking ONNX on CPU. "
            "Install onnxruntime-gpu before proceeding."
        )

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 16000).astype(np.float32)

    logger.info(f"ONNX warmup: {warmup} runs")
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    logger.info(f"ONNX timed runs: {runs}")
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {input_name: dummy})
    t1 = time.perf_counter()

    avg_sec = (t1 - t0) / runs
    avg_ms = avg_sec * 1000.0

    lat_file.write_text(f"{avg_sec:.6f}")
    logger.info(f"ONNX average latency: {avg_ms:.2f} ms")

    # ---- profiling ----
    profile_path = RESULTS_DIR / "onnx_profile.json"
    raw_profile = session.end_profiling()
    os.replace(raw_profile, profile_path)
    logger.info(f"ONNX profiling trace saved → {profile_path}")

    with open(profile_path, "r") as f:
        trace = json.load(f)

    op_time_us = defaultdict(float)
    for ev in trace:
        if ev.get("cat") == "Node":
            op_time_us[ev.get("name", "UNKNOWN")] += float(ev.get("dur", 0))

    op_time_ms = {k: v / 1000.0 for k, v in op_time_us.items()}
    top_ops = sorted(op_time_ms.items(), key=lambda x: x[1], reverse=True)[:30]

    csv_path = RESULTS_DIR / "onnx_ops.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["op_name", "total_time_ms"])
        for op, t in top_ops:
            w.writerow([op, f"{t:.3f}"])

    logger.info(f"ONNX operator CSV saved → {csv_path}")

    # Plot
    ops, times = zip(*top_ops)
    plt.figure(figsize=(10, 6))
    plt.barh(ops[::-1], times[::-1])
    plt.xlabel("Total Time (ms)")
    plt.title("Top 30 ONNX Runtime Operators (CUDA)")
    plt.tight_layout()
    plot_path = RESULTS_DIR / "onnx_top_ops.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"ONNX operator plot saved → {plot_path}")

# ---------------------------
# TensorRT engines benchmarking (via trtexec)
# ---------------------------
# def benchmark_trt_with_trtexec(runs: int, warmup: int, force: bool):
#     trtexec = shutil_which("trtexec")
#     if trtexec is None:
#         logger.warning("trtexec not found on PATH. Skipping TensorRT engine benchmarking (trtexec recommended).")
#         return

#     # find engine files
#     if not TRT_DIR.exists():
#         logger.info("No models/tensorrt directory found. Skipping TRT benchmark.")
#         return

#     engines = list(TRT_DIR.glob("*.plan")) + list(TRT_DIR.glob("*.engine"))
#     if not engines:
#         logger.info("No TensorRT engine files (*.plan / *.engine) found. Skipping TRT benchmark.")
#         return

#     logger.info(f"Found {len(engines)} TRT engine(s). Will try to benchmark each using trtexec.")

#     for eng in engines:
#         # determine precision tag
#         name = eng.name.lower()
#         if "fp16" in name:
#             tag = "trt_fp16"
#             out_lat = RESULTS_DIR / "latency_trt_fp16.txt"
#         elif "mixed" in name:
#             tag = "trt_mixed"
#             out_lat = RESULTS_DIR / "latency_trt_mixed.txt"
#         elif "fp32" in name or "fp_32" in name:
#             tag = "trt_fp32"
#             out_lat = RESULTS_DIR / "latency_trt_fp32.txt"
#         else:
#             tag = f"trt_{eng.stem}"
#             out_lat = RESULTS_DIR / f"latency_{eng.stem}.txt"

#         if file_exists_and_not_forced(out_lat, force):
#             logger.info(f"TRT latency for {eng.name} exists ({out_lat}). Skipping.")
#             continue

#         # call trtexec: repeated runs -> use --iterations; request timing
#         # construct command: ensure we ask trtexec to run N runs and print timing
#         cmd = [
#             trtexec,
#             f"--loadEngine={str(eng)}",
#             f"--warmUp={warmup}",
#             f"--iterations={runs}",
#             "--sync"
#         ]
#         logger.info(f"Running trtexec for engine {eng.name}")
#         try:
#             res = run_subprocess(cmd, capture_output=True, cwd=str(TRT_DIR))
#         except Exception as e:
#             logger.warning(f"trtexec failed for {eng.name}: {e}")
#             continue

#         out = res.stdout + ("\n" + res.stderr if res.stderr else "")
#         # try to parse a latency number from trtexec output
#         latency_sec = parse_trtexec_latency(out)
#         if latency_sec is None:
#             logger.warning(f"Could not parse trtexec latency for {eng.name}. Saving raw output to results for inspection.")
#             raw_path = RESULTS_DIR / f"trtexec_{eng.stem}.txt"
#             raw_path.write_text(out)
#             continue
        
#         latency_ms = latency_sec * 1000
#         write_latency_file(out_lat, latency_ms)
#         logger.info(f"TensorRT engine {eng.name} avg latency ≈ {latency_ms:.6f} ms (from trtexec)")

# ---------------------------
# TensorRT engines benchmarking (via TensorRT API)
# ---------------------------
def benchmark_trt_api(force: bool):
    script = PROJECT_ROOT / "code" / "benchmarking" / "benchmark_trt_engine.py"
    if not script.exists():
        logger.warning("benchmark_trt_engine.py not found; skipping TRT API benchmark")
        return

    engines = TRT_DIR.glob("*.plan")
    for eng in engines:
        name = eng.name.lower()
        if "fp16" in name:
            out = RESULTS_DIR / "latency_trt_fp16.txt"
        elif "fp32" in name:
            out = RESULTS_DIR / "latency_trt_fp32.txt"
        elif "mixed" in name:
            out = RESULTS_DIR / "latency_trt_mixed.txt"
        else:
            continue

        if out.exists() and not force:
            logger.info(f"{out} exists; skipping")
            continue

        run_subprocess([
            sys.executable,
            str(script),
            str(eng),
            str(out)
        ])

# utility: robust parse for trtexec output
def parse_trtexec_latency(output_text: str):
    """
    Attempt multiple heuristics to extract average latency (mili_seconds)
    from trtexec stdout/stderr. Returns mili_seconds (float) or None.
    """
    # common pattern: lines with "mean =  X ms" or "Avg time: X ms" or "mean latency: X ms"
    patterns = [
        r"mean.*?([0-9]*\.?[0-9]+)\s*ms",
        r"avg.*?([0-9]*\.?[0-9]+)\s*ms",
        r"Average.*?([0-9]*\.?[0-9]+)\s*ms",
        r"Mean.*?([0-9]*\.?[0-9]+)\s*ms",
        r"inference.*?([0-9]*\.?[0-9]+)\s*ms",
        r"median.*?([0-9]*\.?[0-9]+)\s*ms",
    ]
    out = output_text.replace("\r", "\n")
    for pat in patterns:
        m = re.search(pat, out, flags=re.IGNORECASE)
        if m:
            try:
                ms = float(m.group(1))
                return ms / 1000.0
            except Exception:
                continue
    # fallback: look for lines like "Timing Trace..." or "Average over X runs: Y ms"
    # fallback: find all floats followed by 'ms' and take the smallest reasonable (median)
    all_ms = [float(x) for x in re.findall(r"([0-9]*\.?[0-9]+)\s*ms", out, flags=re.IGNORECASE)]
    if all_ms:
        # pick median
        arr = np.array(all_ms)
        med = float(np.median(arr))
        # return med / 1000.0 # In mili_seconds
        return med # In milimili_seconds  
    return None

# small helper to find binary in PATH
def shutil_which(name: str):
    from shutil import which
    return which(name)

# ---------------------------
# TensorRT profile parsing (if JSONs exist) - use your parse_trt_profile.py
# ---------------------------
def parse_trt_profiles_if_present():
    parser_script = PROJECT_ROOT / "code" / "tensorrt" / "parse_trt_profile.py"
    if not parser_script.exists():
        logger.debug("parse_trt_profile.py not found; skipping TRTex profile parsing.")
        return
    # look for any trt profile jsons previously generated in models/tensorrt
    candidates = list(TRT_DIR.glob("*.json")) + list(TRT_DIR.glob("*profile*.json")) + list(RESULTS_DIR.glob("*.json"))
    if not candidates:
        logger.debug("No TensorRT profile JSONs found to parse.")
        return

    for p in candidates:
        out_csv = RESULTS_DIR / f"{p.stem}_ops.csv"
        try:
            run_subprocess([sys.executable, str(parser_script), str(p), str(out_csv)])
            logger.info(f"Parsed TRT profile {p} -> {out_csv}")
        except Exception as e:
            logger.warning(f"Failed to parse TRT profile {p}: {e}")

# ---------------------------
# Try running Triton check (calls your triton_check.py)
# ---------------------------
def run_triton_check_if_present():
    script = PROJECT_ROOT / "code" / "triton" / "triton_check.py"
    if not script.exists():
        logger.debug("triton_check.py not found; skipping Triton validation.")
        return
    logger.info("Running Triton validation (triton_check.py)...")
    try:
        res = run_subprocess([sys.executable, str(script)], capture_output=True)
        # triton_check writes latency_triton.json — leave it alone (compare_backends will pick it up)
        logger.info("Triton validation script finished.")
        if res.returncode != 0:
            logger.warning("Triton check returned non-zero exit code; examine triton_check logs")
    except Exception as e:
        logger.warning(f"Triton check failed: {e}")

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    runs = args.runs
    warmup = args.warmup
    force = args.force

    logger.info("Unified benchmark starting")
    logger.info(f"runs={runs} warmup={warmup} force={force}")

    # PyTorch
    try:
        benchmark_pytorch(runs, warmup, force)
    except Exception as e:
        logger.exception(f"PyTorch benchmark failed: {e}")

    # ONNX
    try:
        benchmark_onnx(runs, warmup, force)
    except Exception as e:
        logger.exception(f"ONNX benchmark failed: {e}")

    # # TRT engines via trtexec (best-effort)
    # try:
    #     benchmark_trt_with_trtexec(runs, warmup, force)
    # except Exception as e:
    #     logger.exception(f"Tensorrt (trtexec) benchmarking failed: {e}")

    # TensorRT benchmarking via Python API (preferred)
    try:
        benchmark_trt_api(force)
    except Exception as e:
        logger.exception(f"TensorRT API benchmarking failed: {e}")

    # Parse any TensorRT profile JSONs to CSVs/plots
    try:
        parse_trt_profiles_if_present()
    except Exception as e:
        logger.exception(f"Parsing TRT profiles failed: {e}")

    # Try Triton validation (calls triton_check which writes latency_triton.json)
    try:
        run_triton_check_if_present()
    except Exception as e:
        logger.exception(f"Triton check failed: {e}")

    logger.info("Unified benchmark completed. Now run compare_backends.py to produce cross-backend charts & CSVs.")

if __name__ == "__main__":
    main()