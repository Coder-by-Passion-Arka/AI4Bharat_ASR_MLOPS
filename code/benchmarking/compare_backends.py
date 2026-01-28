#!/usr/bin/env python3
"""
Unified backend benchmarking & comparison.

What it does:
- For each backend (PyTorch, ONNX Runtime, TensorRT [fp16/mixed/fp32], Triton):
    - If latency file exists (e.g. '/results/latency_onnx.txt') in results/, read it.
    - Otherwise attempt a lightweight benchmark and save latency to results/.
- Produce CSV summary and PNG plots.
- normalize units (if value < 1.0 assume seconds -> convert to ms)
- Produce WER plot if results/WER_hi.txt is present.
- Log both with logger and print readable terminal outputs.

Place this file at: code/benchmarking/compare_backends.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt

from code.utils.logger import get_logger

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = get_logger(name="compare_backends", level=10)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Expected latency artifacts
# -----------------------------------------------------------------------------
LATENCY_FILES: Dict[str, Path] = {
    "PyTorch": RESULTS_DIR / "latency_pytorch.txt",
    "ONNX Runtime": RESULTS_DIR / "latency_onnx.txt",
    "TensorRT FP16": RESULTS_DIR / "latency_trt_fp16.txt",
    "TensorRT Mixed": RESULTS_DIR / "latency_trt_mixed.txt",
    "TensorRT FP32": RESULTS_DIR / "latency_trt_fp32.txt",
    "Triton": RESULTS_DIR / "latency_triton.json",
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def read_and_normalize_latency(path: Path) -> Optional[float]:
    """
    Read latency artifact and normalize to milliseconds.

    Supports:
    - .txt  → float (seconds or ms)
    - .json → Triton output

    Rules:
    - < 1.0  → seconds → ms
    - >=1.0 → already ms
    """
    if not path.exists():
        logger.debug(f"Latency file not found: {path}")
        return None

    try:
        # ---------------- Triton JSON ----------------
        if path.suffix == ".json":
            data = json.loads(path.read_text())

            # ✅ FIX: correct Triton keys
            for key in ("avg_ms", "avg_latency_ms", "latency_ms", "p50_ms"):
                if key in data:
                    val = float(data[key])
                    logger.debug(f"Read {key}={val} from {path.name}")
                    return val

            logger.warning(f"No known latency key in Triton JSON: {path}")
            return None

        # ---------------- Plain text ----------------
        raw = path.read_text().strip()
        val = float(raw)

        if val < 1.0:
            logger.debug(f"{path.name}: assuming seconds → ms")
            return val * 1000.0

        return val

    except Exception as e:
        logger.warning(f"Failed to parse latency file {path}: {e}")
        return None


def print_and_log(msg: str):
    print(msg)
    logger.info(msg)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print_and_log("=== Backend Latency Comparison ===")

    rows = []

    for backend, path in LATENCY_FILES.items():
        latency_ms = read_and_normalize_latency(path)

        if latency_ms is None:
            print_and_log(f"[{backend}] ❌ missing or unreadable → {path.name}")
            continue

        rows.append({
            "Backend": backend,
            "Latency_ms": latency_ms,
        })

    if not rows:
        print_and_log("❌ No valid latency files found. Run benchmarking first.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Speedup calculation
    # -------------------------------------------------------------------------
    if "PyTorch" in df["Backend"].values:
        baseline = float(df.loc[df["Backend"] == "PyTorch", "Latency_ms"].iloc[0])
        baseline_name = "PyTorch"
    else:
        baseline = float(df["Latency_ms"].min())
        baseline_name = "Fastest backend"

    df["Speedup_vs_PyTorch"] = baseline / df["Latency_ms"]
    df = df.sort_values("Latency_ms")

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    csv_path = RESULTS_DIR / "backend_latency_comparison.csv"
    df.to_csv(csv_path, index=False)
    print_and_log(f"[Compare] CSV saved → {csv_path}")

    # -------------------------------------------------------------------------
    # Plot latency bar chart
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    plt.bar(df["Backend"], df["Latency_ms"])
    plt.ylabel("Latency (ms)")
    plt.title("Backend Latency Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "backend_latency_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print_and_log(f"[Compare] Plot saved → {plot_path}")

    # -------------------------------------------------------------------------
    # Optional WER plot
    # -------------------------------------------------------------------------
    wer_file = RESULTS_DIR / "WER_hi.txt"
    if wer_file.exists():
        try:
            for line in wer_file.read_text().splitlines():
                if "WER" in line:
                    wer = float(line.split(":")[-1].strip())
                    plt.figure(figsize=(4, 3))
                    plt.bar(["Hindi"], [wer])
                    plt.ylabel("WER")
                    plt.title("ASR Accuracy (Hindi)")
                    plt.tight_layout()

                    wer_plot = RESULTS_DIR / "wer_hi.png"
                    plt.savefig(wer_plot, dpi=150)
                    plt.close()
                    print_and_log(f"[Compare] WER plot saved → {wer_plot}")
                    break
        except Exception:
            logger.exception("Failed to parse WER file")

    # -------------------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------------------
    print("\n=== Backend Latency Summary (ms) ===\n")
    print(
        df[["Backend", "Latency_ms", "Speedup_vs_PyTorch"]]
        .to_string(index=False, float_format="{:,.3f}".format)
    )

    print_and_log(f"\nBaseline used for speedup: {baseline_name}")
    print_and_log("=== Comparison complete ===")


if __name__ == "__main__":
    main()

# from __future__ import annotations
# import os
# import sys
# import time
# import json
# import math
# import numpy as np
# import pandas as pd
# from pathlib import Path
# import matplotlib.pyplot as plt
# from typing import Optional, Dict, Tuple, List

# # Use the project's logger (console + file)
# from code.utils.logger import get_logger

# logger = get_logger(name="compare_backends", level=10)  # DEBUG

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = PROJECT_ROOT / "results"
# MODELS_DIR = PROJECT_ROOT / "models"
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LATENCY_FILES = {
#     "PyTorch": RESULTS_DIR / "latency_pytorch.txt",
#     "ONNX Runtime": RESULTS_DIR / "latency_onnx.txt",
#     "TensorRT FP16": RESULTS_DIR / "latency_trt_fp16.txt",
#     "TensorRT Mixed": RESULTS_DIR / "latency_trt_mixed.txt",
#     "TensorRT FP32": RESULTS_DIR / "latency_trt_fp32.txt",
#     "Triton": RESULTS_DIR / "latency_triton.txt",
# }

# # constants
# DEFAULT_RUNS = 20
# DEFAULT_WARMUP = 5

# def read_and_normalize(path: Path):
#     """
#     Read latency number from path. Heuristic normalization:
#       - if value < 1.0 -> assume seconds -> convert to ms
#       - otherwise assume value is already milliseconds
#     Returns float (ms) or None if file missing / unreadable.
#     """
#     if not path.exists():
#         return None
#     try:
#         raw = path.read_text().strip()
#         v = float(raw)
#         if v < 1.0:
#             # value looks like seconds -> convert to ms
#             return v * 1000.0
#         return v
#     except Exception:
#         return None


# def _read_latency_file(path: Path) -> Optional[float]:
#     if not path.exists():
#         return None
#     try:
#         txt = path.read_text().strip()
#         val = float(txt)
#         logger.debug(f"Read latency file {path.name}: {val}")
#         return val
#     except Exception as e:
#         logger.warning(f"Failed reading latency file {path}: {e}")
#         return None

# def _write_latency_file(path: Path, seconds: float):
#     path.write_text(f"{seconds:.6f}")
#     logger.info(f"Saved latency → {path}")

# def _print_and_log(msg: str, level="info"):
#     # print raw for engineer friendly view, then log
#     print(msg)
#     if level == "info":
#         logger.info(msg)
#     elif level == "debug":
#         logger.debug(msg)
#     elif level == "warning":
#         logger.warning(msg)
#     elif level == "error":
#         logger.error(msg)
#     else:
#         logger.info(msg)

# # -------------------------
# # PyTorch Benchmark
# # -------------------------
# def benchmark_pytorch_if_needed(
#     model_name: str = "ai4bharat/indicwav2vec-hindi",
#     out_path: Path = LAT_PYTORCH,
#     runs: int = DEFAULT_RUNS, warmup: int = DEFAULT_WARMUP
# ) -> Optional[float]:
#     existing = _read_latency_file(out_path)
#     if existing is not None:
#         _print_and_log(f"[PyTorch] Using existing latency: {existing:.6f} s", "info")
#         return existing

#     _print_and_log("[PyTorch] Running lightweight benchmark (this will load the model)...", "info")
#     try:
#         import torch
#         from transformers import AutoModelForCTC
#     except Exception as e:
#         logger.exception("PyTorch or transformers not available; skipping PyTorch benchmark")
#         return None

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     _print_and_log(f"[PyTorch] Device: {device}", "debug")
#     try:
#         model = AutoModelForCTC.from_pretrained(model_name).to(device)
#         model.eval()
#     except Exception as e:
#         logger.exception("Failed loading PyTorch model; skipping PyTorch benchmark")
#         return None

#     dummy = torch.randn(1, 16000, dtype=torch.float32, device=device)
#     # warmup
#     with torch.no_grad():
#         for _ in range(warmup):
#             _ = model(dummy).logits
#     # timed runs
#     times = []
#     with torch.no_grad():
#         for _ in range(runs):
#             t0 = time.perf_counter()
#             _ = model(dummy).logits
#             t1 = time.perf_counter()
#             times.append(t1 - t0)
#     avg = float(np.mean(times))
#     _write_latency_file(out_path, avg)
#     _print_and_log(f"[PyTorch] Average latency: {avg:.6f} s ({avg*1000:.2f} ms)", "info")
#     return avg

# # -------------------------
# # ONNX Runtime Benchmark
# # -------------------------
# def benchmark_onnx_if_needed(onnx_path: Optional[Path] = None,
#                              out_path: Path = LAT_ONNX,
#                              runs: int = DEFAULT_RUNS, warmup: int = DEFAULT_WARMUP) -> Optional[float]:
#     existing = _read_latency_file(out_path)
#     if existing is not None:
#         _print_and_log(f"[ONNX] Using existing latency: {existing:.6f} s", "info")
#         return existing

#     try:
#         import onnxruntime as ort
#     except Exception:
#         logger.exception("onnxruntime not installed; skipping ONNX benchmark")
#         return None

#     if onnx_path is None:
#         onnx_cand = MODELS_DIR / "onnx" / "wav2vec2_optimized.onnx"
#         if onnx_cand.exists():
#             onnx_path = onnx_cand
#         else:
#             onnx_cand = MODELS_DIR / "onnx" / "wav2vec2.onnx"
#             if onnx_cand.exists():
#                 onnx_path = onnx_cand
#             else:
#                 logger.warning("No ONNX model found under models/onnx; skipping ONNX benchmark")
#                 return None

#     _print_and_log(f"[ONNX] Using ONNX model: {onnx_path}", "info")

#     # Session options with profiling disabled here (small run)
#     so = ort.SessionOptions()
#     # try to use CUDA if available
#     providers = []
#     try:
#         available = ort.get_available_providers()
#         providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
#     except Exception:
#         providers = ["CPUExecutionProvider"]

#     sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
#     if "CUDAExecutionProvider" not in sess.get_providers():
#         logger.warning("CUDAExecutionProvider not active – inference will run on CPU (slower)")

#     # prepare dummy input
#     import numpy as _np
#     dummy = _np.random.randn(1, 16000).astype(_np.float32)
#     input_name = sess.get_inputs()[0].name

#     # warmup
#     for _ in range(warmup):
#         sess.run(None, {input_name: dummy})

#     times = []
#     for _ in range(runs):
#         t0 = time.perf_counter()
#         sess.run(None, {input_name: dummy})
#         t1 = time.perf_counter()
#         times.append(t1 - t0)

#     avg = float(np.mean(times))
#     _write_latency_file(out_path, avg)
#     _print_and_log(f"[ONNX] Average latency: {avg:.6f} s ({avg*1000:.2f} ms)", "info")

#     # try to produce operator CSV & small plot if session profiling available elsewhere
#     try:
#         # If ORT profiling existed you'd parse it; here we skip deep profiling
#         pass
#     except Exception:
#         logger.debug("ONNX profiling extras skipped")

#     return avg

# # -------------------------
# # TensorRT Benchmark per-engine
# # -------------------------
# def benchmark_trt_engine(engine_path: Path, out_path: Path, runs: int = DEFAULT_RUNS, warmup: int = DEFAULT_WARMUP) -> Optional[float]:
#     existing = _read_latency_file(out_path)
#     if existing is not None:
#         _print_and_log(f"[TensorRT:{engine_path.name}] Using existing latency: {existing:.6f} s", "info")
#         return existing

#     try:
#         import tensorrt as trt
#         import pycuda.driver as cuda
#         import pycuda.autoinit  # noqa: F401
#     except Exception:
#         logger.exception("tensorrt/pycuda not available; skipping TensorRT benchmark")
#         return None

#     _print_and_log(f"[TensorRT] Benchmarking engine: {engine_path}", "info")
#     try:
#         TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#         with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
#             engine = rt.deserialize_cuda_engine(f.read())
#     except Exception as e:
#         logger.exception(f"Failed to deserialize engine {engine_path}: {e}")
#         return None

#     try:
#         context = engine.create_execution_context()
#         # pick input size from engine binding if possible
#         # fallback to 1x16000
#         # Create dummy host array sized by engine bindings if single binding
#         import numpy as _np
#         # Try to figure binding size (best-effort)
#         input_shape = (1, 16000)
#         # allocate device buffer for input; assume float32
#         h_input = _np.random.randn(*input_shape).astype(_np.float32)
#         d_input = cuda.mem_alloc(h_input.nbytes)
#         cuda.memcpy_htod(d_input, h_input)
#         stream = cuda.Stream()

#         # warmup
#         for _ in range(warmup):
#             # For modern TRT, use execute_async_v2 with device pointers in list
#             context.execute_async_v2([int(d_input)], stream.handle)
#             stream.synchronize()

#         times = []
#         for _ in range(runs):
#             t0 = time.perf_counter()
#             context.execute_async_v2([int(d_input)], stream.handle)
#             stream.synchronize()
#             t1 = time.perf_counter()
#             times.append(t1 - t0)

#         avg = float(np.mean(times))
#         _write_latency_file(out_path, avg)
#         _print_and_log(f"[TensorRT:{engine_path.name}] Average latency: {avg:.6f} s ({avg*1000:.2f} ms)", "info")
#         return avg
#     except Exception as e:
#         logger.exception("TensorRT execution error")
#         return None

# def benchmark_trt_if_needed(trt_dir: Optional[Path] = None):
#     if trt_dir is None:
#         trt_dir = MODELS_DIR / "tensorrt"
#     if not trt_dir.exists():
#         logger.debug("No TensorRT models folder; skipping TRT benchmarks")
#         return {}

#     results = {}
#     # find candidate plan files
#     for p in trt_dir.glob("*.plan"):
#         name = p.name.lower()
#         if "fp16" in name:
#             out = LAT_TRT_FP16
#             results["TensorRT FP16"] = benchmark_trt_engine(p, out)
#         elif "mixed" in name:
#             out = LAT_TRT_MIXED
#             results["TensorRT Mixed"] = benchmark_trt_engine(p, out)
#         elif "fp32" in name:
#             out = LAT_TRT_FP32
#             results["TensorRT FP32"] = benchmark_trt_engine(p, out)
#         else:
#             # unknown: map to fp32 fallback
#             out = LAT_TRT_FP32
#             results[f"TensorRT ({p.name})"] = benchmark_trt_engine(p, out)
#     return results

# # -------------------------
# # Triton Benchmark
# # -------------------------
# def benchmark_triton_if_needed(triton_url: str = "localhost:8000", out_path: Path = LAT_TRITON,
#                               runs: int = DEFAULT_RUNS, warmup: int = DEFAULT_WARMUP) -> Optional[float]:
#     existing = _read_latency_file(out_path)
#     if existing is not None:
#         _print_and_log(f"[Triton] Using existing latency: {existing:.6f} s", "info")
#         return existing

#     try:
#         import tritonclient.http as httpclient
#     except Exception:
#         logger.debug("tritonclient not installed; skipping Triton benchmark")
#         return None

#     _print_and_log("[Triton] Running Triton latency check...", "info")
#     try:
#         client = httpclient.InferenceServerClient(url=triton_url)
#     except Exception as e:
#         logger.exception("Failed to connect to Triton server")
#         return None

#     # Model name and IO names as used in deployment
#     MODEL_NAME = "wav2vec2"
#     INPUT_NAME = "input_values"
#     OUTPUT_NAME = "logits"

#     # Build dummy audio
#     import numpy as _np
#     audio = _np.random.randn(1, 16000).astype(_np.float32)
#     inputs = [httpclient.InferInput(INPUT_NAME, audio.shape, "FP32")]
#     inputs[0].set_data_from_numpy(audio)
#     outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]

#     # warmup
#     for _ in range(warmup):
#         client.infer(MODEL_NAME, inputs, outputs=outputs)

#     latencies = []
#     for _ in range(runs):
#         t0 = time.perf_counter()
#         client.infer(MODEL_NAME, inputs, outputs=outputs)
#         t1 = time.perf_counter()
#         latencies.append(t1 - t0)

#     avg = float(np.mean(latencies))
#     _write_latency_file(out_path, avg)
#     _print_and_log(f"[Triton] Avg latency: {avg:.6f} s ({avg*1000:.2f} ms)", "info")
#     return avg

# # -------------------------
# # Main orchestrator
# # -------------------------
# def main(run_all: bool = True):
#     _print_and_log("=== Backend Benchmark & Comparison ===", "info")

#     # 1) PyTorch
#     pytorch_s = benchmark_pytorch_if_needed()

#     # 2) ONNX Runtime
#     onnx_s = benchmark_onnx_if_needed()

#     # 3) TensorRT (check engines and benchmark)
#     trt_results = benchmark_trt_if_needed()

#     # 4) Triton (optional)
#     triton_s = benchmark_triton_if_needed()

#     # Collect latencies into mapping (seconds)
#     lat_map = {
#         "PyTorch": pytorch_s,
#         "ONNX Runtime": onnx_s,
#         "TensorRT FP16": _read_latency_file(LAT_TRT_FP16),
#         "TensorRT Mixed": _read_latency_file(LAT_TRT_MIXED),
#         "TensorRT FP32": _read_latency_file(LAT_TRT_FP32),
#         "Triton": _read_latency_file(LAT_TRITON),
#     }

#     # Filter out None
#     entries = {k: v for k, v in lat_map.items() if v is not None}
#     if not entries:
#         _print_and_log("No latency data available. Run individual benchmark steps first.", "error")
#         sys.exit(1)

#     # Convert seconds -> ms for display
#     rows = []
#     baseline_sec = entries.get("PyTorch", None)
#     for k, sec in entries.items():
#         ms = sec * 1000.0
#         speedup = (baseline_sec / sec) if (baseline_sec and sec and sec > 0) else None
#         rows.append({"Backend": k, "Latency_ms": ms, "Latency_s": sec, "Speedup_vs_PyTorch": speedup})

#     df = pd.DataFrame(rows).sort_values("Latency_ms")
#     csv_out = RESULTS_DIR / "backend_latency_comparison.csv"
#     df.to_csv(csv_out, index=False)
#     _print_and_log(f"[Compare] Saved backend latency table → {csv_out}", "info")

#     # Plot bar chart (ms)
#     plt.figure(figsize=(9, 4))
#     plt.bar(df["Backend"], df["Latency_ms"])
#     plt.ylabel("Latency (ms)")
#     plt.title("Backend Latency Comparison")
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plot_path = RESULTS_DIR / "backend_latency_comparison.png"
#     plt.savefig(plot_path, dpi=150)
#     plt.close()
#     _print_and_log(f"[Compare] Saved latency plot → {plot_path}", "info")

#     # WER plot (if available)
#     wer_file = RESULTS_DIR / "WER_hi.txt"
#     if wer_file.exists():
#         try:
#             for ln in wer_file.read_text().splitlines():
#                 if "WER" in ln:
#                     val = float(ln.split(":")[-1].strip())
#                     plt.figure(figsize=(4, 3))
#                     plt.bar(["Hindi WER"], [val])
#                     plt.ylabel("WER")
#                     plt.title("ASR Accuracy (Hindi)")
#                     plt.tight_layout()
#                     wer_plot = RESULTS_DIR / "wer_hi.png"
#                     plt.savefig(wer_plot, dpi=150)
#                     plt.close()
#                     _print_and_log(f"[Compare] Saved WER plot → {wer_plot}", "info")
#                     break
#         except Exception:
#             logger.exception("Failed to parse WER file")

#     # Print final table nicely to terminal (engineer-friendly)
#     _print_and_log("\n=== Backend Latency Summary ===", "info")
#     print(df[["Backend", "Latency_ms", "Speedup_vs_PyTorch"]].to_string(index=False))
#     _print_and_log("=== Done ===", "info")

# if __name__ == "__main__":
#     main()