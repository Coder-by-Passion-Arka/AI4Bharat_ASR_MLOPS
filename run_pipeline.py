#!/usr/bin/env python3
"""
AI4Bharat ASR Optimization – Local Execution Script (UPDATED)
- More verbose, timestamped progress logging
- Prefer Python TensorRT API builder (build_engine_py.build_trt_engines) and fall back to trtexec scripts
- Print durations for each step
"""

import os
import re
import sys
import time
import glob
import torch
import shutil
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

# PROJECT layout
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# -------------------------------------------
# Paths
# -------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TRITON_REPO = CODE_DIR / "triton" / "model_repository"

for d in [CODE_DIR, MODELS_DIR, RESULTS_DIR, TRITON_REPO]:
    d.mkdir(parents=True, exist_ok=True)

import importlib.util
_logger_spec = importlib.util.spec_from_file_location("logger", CODE_DIR / "utils" / "logger.py")
_logger_module = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(_logger_module)
get_logger = _logger_module.get_logger

logger = get_logger()

logger.info("Pipeline started")
logger.warning("This is a warning")
logger.error("Something went wrong")

# Ensure directories exist
for d in [CODE_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def ts():
    # return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("- "* 30)

def run_cmd(cmd, cwd=None, env=None, show_cmd=True):
    """
    Run a command as a subprocess while ensuring:
    - Project root is on PYTHONPATH
    - Logs are visible in terminal
    """

    if show_cmd:
        logger.info(f"[CMD] {' '.join(cmd)} (cwd={cwd})")

    # --- CRITICAL FIX ---
    # Ensure project root is available to subprocesses
    run_env = (env or os.environ.copy()).copy()
    run_env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        + os.pathsep
        + run_env.get("PYTHONPATH", "")
    )

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=run_env,
    )
    elapsed = time.time() - start

    logger.info(f"[CMD] finished (rc={result.returncode}) in {elapsed:.2f}s")

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {' '.join(cmd)}"
        )

def check_gpu():
    print(f"{ts()} Checking GPU availability...")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            print("GPU available but get_device_name failed.")
    else:
        print("WARNING: CUDA not available. GPU optimizations will not work.")

def ensure_dirs():
    for d in (MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Pipeline steps
# --------------------------------------------------

def timed_step(name, fn, *args, **kwargs):
    print(f"\n{ts()} [STEP] {name} -- START")
    t0 = time.time()
    try:
        res = fn(*args, **kwargs)
        status = "OK"
    except Exception as e:
        res = e
        status = "FAILED"
    t1 = time.time()
    print(f"{ts()} [STEP] {name} -- {status} (elapsed: {t1-t0:.2f}s)")
    if status == "FAILED":
        raise res
    return res

def run_profiler():
    run_cmd([sys.executable, str(CODE_DIR / "profiling" / "pytorch_profiler.py")])

def export_onnx():
    run_cmd([sys.executable, str(CODE_DIR / "onnx_export" / "export_to_onnx.py")])
    print("Models folder listing:")
    run_cmd(["ls", "-lh", str(MODELS_DIR)])

def optimize_onnx():
    run_cmd([sys.executable, str(CODE_DIR / "onnx_optim" / "optimize_onnx.py")])
    print("Models folder listing:")
    run_cmd(["ls", "-lh", str(MODELS_DIR)])

def benchmark_latency():
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "benchmark_latency.py", "--force")])

def compute_wer():
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "compute_wer.py")])

def build_tensorrt_engine():
    build_dir = CODE_DIR / "tensorrt"
    onnx_path = MODELS_DIR / "onnx" / "wav2vec2_optimized.onnx"
    out_dir = MODELS_DIR / "tensorrt"
    out_dir.mkdir(parents=True, exist_ok=True)

    trtexec = shutil.which("trtexec")
    if trtexec:
        # test whether trtexec runs correctly
        try:
            ret = subprocess.run([trtexec, "--version"], capture_output=True, text=True, timeout=10)
            if ret.returncode == 0:
                print("[TRT] trtexec is available and seems functional. Using trtexec for build (if scripts exist).")
                # call existing scripts (they call trtexec internally)
                fp16 = build_dir / "build_engine_fp16.sh"
                mixed = build_dir / "build_engine_mixed.sh"
                if fp16.exists(): run_cmd(["bash", str(fp16)], cwd=str(build_dir))
                if mixed.exists(): run_cmd(["bash", str(mixed)], cwd=str(build_dir))
                return
            else:
                print("[TRT] trtexec exists but returned non-zero --version. Will fallback to Python API builder.")
        except Exception as e:
            print("[TRT] trtexec check failed:", e)
            print("[TRT] Will fallback to Python API builder.")

    # Fallback: use Python builder directly
    builder_py = CODE_DIR / "tensorrt" / "build_engine_py.py"
    if not builder_py.exists():
        raise FileNotFoundError("build_engine_py.py not found in code/tensorrt")
    for precision in ("fp16", "mixed", "fp32"):
        out_name = f"wav2vec2_trt_{precision}.plan"
        out_path = out_dir / out_name
        cmd = [
            sys.executable,
            str(builder_py),
            "--onnx", str(onnx_path),
            "--out_dir", str(out_dir),
            "--precision", precision,
            # you can tune min/opt/max here if desired
        ]
        try:
            print(f"[TRT] Building {precision} engine via Python API...")
            run_cmd(cmd, cwd=str(build_dir))
        except Exception as e:
            print(f"[TRT] Build {precision} failed: {e}")
            # continue trying other precisions

def parse_trt_profiles():
    p1 = MODELS_DIR / "tensorrt" / "trt_fp16_profile.json"
    p2 = MODELS_DIR / "tensorrt" / "trt_mixed_profile.json"
    parser = CODE_DIR / "tensorrt" / "parse_trt_profile.py"
    if p1.exists():
        run_cmd([sys.executable, str(parser), str(p1), str(RESULTS_DIR / "trt_fp16_ops.csv")])
    else:
        print(f"{ts()} [TensorRT] No fp16 profile found; skipping.")
    if p2.exists():
        run_cmd([sys.executable, str(parser), str(p2), str(RESULTS_DIR / "trt_mixed_ops.csv")])
    else:
        print(f"{ts()} [TensorRT] No mixed profile found; skipping.")

def compare_results():
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "compare_backends.py")])
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "compare_ops.py")])

def _choose_trt_engine(project_root: Path):
    TRT_DIR = PROJECT_ROOT / "models" / "tensorrt"
    TRITON_REPO = PROJECT_ROOT / "triton" / "wav2vec2" / "1"

    TRITON_REPO.mkdir(parents=True, exist_ok=True)

    # Choose which engine Triton should serve (recommend FP16 or mixed)
    ENGINE_SRC = TRT_DIR / "wav2vec2_optimized_trt_fp16.plan"
    ENGINE_DST = TRITON_REPO / "model.plan"

    if not ENGINE_SRC.exists():
        raise RuntimeError(f"TensorRT engine missing: {ENGINE_SRC}")

    shutil.copyfile(ENGINE_SRC, ENGINE_DST)

    # sanity check
    if ENGINE_DST.stat().st_size < 1024:
        raise RuntimeError("Copied Triton model.plan is empty or corrupted")

    print(f"[Pipeline] Triton model updated → {ENGINE_DST}")
    candidates = {
        "fp16": [TRT_DIR / "wav2vec2_fp16.plan", TRT_DIR / "wav2vec2_fp16.engine", TRT_DIR / "wav2vec2_fp16.trt"],
        "mixed": [TRT_DIR / "wav2vec2_mixed.plan", TRT_DIR / "wav2vec2_mixed.engine"],
        "fp32": [TRT_DIR / "wav2vec2_fp32.plan", TRT_DIR / "wav2vec2_fp32.engine"],
    }
    found = {}
    for k, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[k] = p
                break
    return found

# Old Version
# def select_and_deploy_trt_to_triton(project_root: Path):
    print(f"{ts()} [Triton] Attempting to select a TRT engine for Triton model_repository...")
    found = _choose_trt_engine(project_root)
    if not found:
        print(f"{ts()} [Triton] No TensorRT engine files found in models/tensorrt/. Skipping Triton deployment.")
        return

    lat_files = {
        "fp16": project_root / "results" / "latency_trt_fp16.txt",
        "mixed": project_root / "results" / "latency_trt_mixed.txt",
        "fp32": project_root / "results" / "latency_trt_fp32.txt",
    }
    scores = {}
    for k, path in found.items():
        lat_path = lat_files.get(k)
        if lat_path and lat_path.exists():
            try:
                scores[k] = float(lat_path.read_text().strip())
            except Exception:
                scores[k] = float("inf")
        else:
            scores[k] = float("inf")

    chosen = min(scores, key=scores.get)
    engine_path = found[chosen]
    print(f"{ts()} [Triton] Chosen engine: {chosen} -> {engine_path}")

    triton_root = project_root / "triton" / "model_repository" / "wav2vec2"
    version_dir = triton_root / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    dst_engine = version_dir / "model.plan"
    shutil.copyfile(str(engine_path), str(dst_engine))
    print(f"{ts()} [Triton] Copied engine to {dst_engine}")

    cfg_path = triton_root / "config.pbtxt"
    cfg = f"""
            name: "wav2vec2"
            backend: "tensorrt"
            max_batch_size: 8

            input [
            {{
            name: "input_values"
            data_type: TYPE_FP32
            dims: [-1, -1]
            }}
            ]

            output [
            {{
            name: "logits"
            data_type: TYPE_FP32
            dims: [-1, -1]
            }}
            ]

            instance_group [
            {{
            kind: KIND_GPU
            count: 1
            }}
            ]
        """
    cfg_path.write_text(cfg.strip())
    print(f"{ts()} [Triton] Wrote config to {cfg_path}")

    triton_exe = shutil.which("tritonserver")
    if triton_exe is None:
        print(f"{ts()} [Triton] tritonserver not found on PATH. To run Triton manually:\n  tritonserver --model-repository={project_root/'triton'/'model_repository'} --strict-model-config=false")
        return

    print(f"{ts()} [Triton] Starting tritonserver (this will block until terminated)...")
    run_cmd([triton_exe, f"--model-repository={project_root/'triton'/'model_repository'}", "--strict-model-config=false"])

# New Version 
def _discover_trt_engines(trt_dir: Path):
    # return dict precision -> path
    found = {}
    for p in trt_dir.glob("*.plan"):
        name = p.name.lower()
        # infer precision from filename if possible
        if "fp16" in name:
            found.setdefault("fp16", []).append(p)
        elif "mixed" in name:
            found.setdefault("mixed", []).append(p)
        elif "fp32" in name:
            found.setdefault("fp32", []).append(p)
        else:
            # fallback: treat unknown as fp32 candidate
            found.setdefault("fp32", []).append(p)
    return found

def select_and_deploy_trt_to_triton(project_root: Path):
    print("[Triton] Attempting to select a TRT engine for Triton model_repository...")
    TRT_DIR = project_root / "models" / "tensorrt"
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    found = _discover_trt_engines(TRT_DIR)

    if not found:
        print("[Triton] No TensorRT engine files found in models/tensorrt/. Skipping Triton deployment.")
        return

    # read latency files if present
    lat_files = {
        "fp16": project_root / "results" / "latency_trt_fp16.txt",
        "mixed": project_root / "results" / "latency_trt_mixed.txt",
        "fp32": project_root / "results" / "latency_trt_fp32.txt",
    }
    scores = {}
    # choose best file per precision (smallest size) if multiple
    chosen_per_precision = {}
    for prec, paths in found.items():
        # pick smallest file (heuristic)
        chosen = min(paths, key=lambda p: p.stat().st_size) if paths else None
        chosen_per_precision[prec] = chosen
        # latency score (lower better), default inf
        pf = lat_files.get(prec)
        if pf and pf.exists():
            try:
                scores[prec] = float(pf.read_text().strip())
            except Exception:
                scores[prec] = float("inf")
        else:
            scores[prec] = float("inf")

    # If any have real latency, pick smallest; else prefer fp16->mixed->fp32
    real_lat = {k: v for k, v in scores.items() if v not in (None, float("inf"))}
    if real_lat:
        chosen_prec = min(real_lat, key=real_lat.get)
    else:
        for pref in ("fp16", "mixed", "fp32"):
            if chosen_per_precision.get(pref) is not None:
                chosen_prec = pref
                break

    if chosen_prec is None:
        print("[Triton] Unable to find a suitable engine to deploy.")
        return

    chosen_path = chosen_per_precision[chosen_prec]
    print(f"[Triton] Selected engine: {chosen_prec} -> {chosen_path}")

    # prepare model repository
    triton_model_dir = project_root / "triton" / "model_repository" / "wav2vec2"
    version_dir = triton_model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    dst_engine = version_dir / "model.plan"
    shutil.copyfile(str(chosen_path), str(dst_engine))
    print(f"[Triton] Copied engine to {dst_engine}")

    # write minimal config.pbtxt (same as your existing)
    cfg_path = triton_model_dir / "config.pbtxt"
    cfg = f"""
name: "wav2vec2"
backend: "tensorrt"
max_batch_size: 8

input [
{{
  name: "input_values"
  data_type: TYPE_FP32
  dims: [-1, -1]
}}
]

output [
{{
  name: "logits"
  data_type: TYPE_FP32
  dims: [-1, -1]
}}
]

instance_group [
{{
  kind: KIND_GPU
  count: 1
}}
]
"""
    cfg_path.write_text(cfg.strip())
    print(f"[Triton] Wrote config to {cfg_path}")

    triton_exe = shutil.which("tritonserver")
    if triton_exe:
        print("[Triton] Starting tritonserver (will block). Run manually if you prefer.")
        run_cmd([triton_exe, f"--model-repository={project_root/'triton'/'model_repository'}", "--strict-model-config=false"])
    else:
        print("[Triton] tritonserver not found on PATH. To run manually:")
        print(f"  tritonserver --model-repository={project_root/'triton'/'model_repository'} --strict-model-config=false")

def _choose_netron_candidate(paths):
    for p in paths:
        if p.exists():
            return str(p)
    return None


def launch_netron(model_path: str, port: int, label: str):
    if not shutil.which("netron"):
        print(f"{ts()} [Netron] netron not installed (pip install netron). Skipping Netron for {label}")
        return
    if not Path(model_path).exists():
        print(f"{ts()} [Netron] Model not found for {label}: {model_path}; skipping.")
        return
    print(f"{ts()} [Netron] Launching {label} on port {port}...")
    subprocess.Popen(["netron", model_path, "--port", str(port), "--host", "0.0.0.0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def launch_all_netron():
    p_torch = _choose_netron_candidate([
        PROJECT_ROOT / "models" / "pytorch" / "wav2vec2.pt",
        PROJECT_ROOT / "models" / "pytorch" / "wav2vec2_state_dict.pt"
    ])
    p_onnx = _choose_netron_candidate([
        PROJECT_ROOT / "models" / "onnx" / "wav2vec2.onnx",
        PROJECT_ROOT / "models" / "onnx" / "wav2vec2_optimized.onnx"
    ])
    p_trt = _choose_netron_candidate([
        PROJECT_ROOT / "models" / "tensorrt" / "wav2vec2_fp16.plan",
        PROJECT_ROOT / "models" / "tensorrt" / "wav2vec2_fp16.engine",
        PROJECT_ROOT / "models" / "tensorrt" / "wav2vec2_mixed.plan",
        PROJECT_ROOT / "models" / "tensorrt" / "wav2vec2_fp32.plan",
    ])

    if p_torch:
        launch_netron(p_torch, 8081, "PyTorch")
    if p_onnx:
        launch_netron(p_onnx, 8082, "ONNX")
    if p_trt:
        launch_netron(p_trt, 8083, "TensorRT")

    time.sleep(2)
    print(f"\n{ts()} [Netron] Graph viewers (if launched):")
    if p_torch:
        print("  PyTorch → http://localhost:8081")
    if p_onnx:
        print("  ONNX    → http://localhost:8082")
    if p_trt:
        print("  TRT     → http://localhost:8083")

# Old Triton Starting Code
# def start_triton_server(project_root: Path):
#     triton_exe = shutil.which("tritonserver")
#     if not triton_exe:
#         logger.warning("tritonserver not found. Skipping auto-start.")
#         return None

#     cmd = [
#         triton_exe,
#         f"--model-repository={project_root/'triton'/'model_repository'}",
#         "--strict-model-config=false"
#     ]

#     logger.info("[Triton] Starting Triton server...")
#     return subprocess.Popen(cmd)

# New Triton Starting Code
# -----------------------------------------------
# Triton deployment (CRITICAL FIX)
# -----------------------------------------------
def deploy_best_trt_to_triton():
    trt_dir = MODELS_DIR / "tensorrt"
    model_dir = TRITON_REPO / "wav2vec2"
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    preferred = trt_dir / "wav2vec2_optimized_trt_mixed.plan"
    fallback = trt_dir / "wav2vec2_optimized_trt_fp16.plan"

    if preferred.exists():
        engine = preferred
        print("[Triton] Using MIXED precision engine")
    elif fallback.exists():
        engine = fallback
        print("[Triton] Mixed not found, using FP16 engine")
    else:
        raise RuntimeError("No TensorRT engine found for Triton deployment")

    dst = version_dir / "model.plan"
    shutil.copyfile(engine, dst)

    if dst.stat().st_size < 1024:
        raise RuntimeError("Copied model.plan is empty or corrupted")

    print(f"[Triton] Engine deployed → {dst}")

def start_triton_server():
    model_repo = TRITON_REPO
    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "-p", "8000:8000",
        "-p", "8001:8001",
        "-p", "8002:8002",
        "-v", f"{model_repo}:/models",
        "nvcr.io/nvidia/tritonserver:24.12-py3",
        "tritonserver",
        "--model-repository=/models",
        "--strict-model-config=true"
    ]

    print("\n[Triton] Starting Triton Inference Server")
    print("👉 http://localhost:8000")
    subprocess.Popen(cmd)
    time.sleep(12)

def run_triton_check():
    run_cmd([sys.executable, CODE_DIR / "triton" / "triton_check.py"])

def compare_results():
    run_cmd([sys.executable, CODE_DIR / "benchmarking" / "compare_backends.py"])


def main():
    print("=" * 60)
    print("AI4Bharat ASR Optimization – Local Pipeline")
    print("=" * 60)
    ensure_dirs()
    check_gpu()

    steps = [
        ("PyTorch Profiling", run_profiler),
        ("Export ONNX", export_onnx),
        ("Optimize ONNX", optimize_onnx),
        ("Benchmark ONNX Runtime", benchmark_latency),
        ("Compute WER", compute_wer),
        ("Build TensorRT Engines", build_tensorrt_engine),
        # ("Deploy best TRT to Triton", lambda: select_and_deploy_trt_to_triton(PROJECT_ROOT)),
        ("Deploy TensorRT → Triton", deploy_best_trt_to_triton),
        ("Parse TRT Profiles", parse_trt_profiles),
        ("Compare Results", compare_results),
        ("Starting Triton Server...", start_triton_server
        # (PROJECT_ROOT)
        ),
        ("Triton Health Check", run_triton_check),
        ("Compare Backends", compare_results),
    ]

    for name, fn in steps:
        try:
            timed_step(name, fn)
        except Exception as e:
            print(f"{ts()} [WARN] Step '{name}' failed: {e}\nContinuing with next steps...")

    print(f"\n{ts()} [INFO] Launching Netron graph viewers (if netron installed)...")
    launch_all_netron()

    print(f"\n{ts()} [STEP] Triton Server Validation & Benchmark")

    print(f"\n{ts()} Pipeline finished (some steps may have been skipped or failed).")

    # try:
    #     run_triton_check()
    # except Exception as e:
    #     print("[STEP] Triton validation failed:", e)

if __name__ == "__main__":
    main()
