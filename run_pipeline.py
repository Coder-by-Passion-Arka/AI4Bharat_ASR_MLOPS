#!/usr/bin/env python3
"""
AI4Bharat ASR Optimization – Local Execution Script
"""

# import os
# import sys
# import subprocess
# import torch
# import shutil

# # ---------------------------------------------------------------------
# # Path resolution (CRITICAL FIX)
# # ---------------------------------------------------------------------

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# CODE_DIR = os.path.join(PROJECT_ROOT, "code")
# MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# # ---------------------------------------------------------------------
# # Utility helpers
# # ---------------------------------------------------------------------

# def run_cmd(cmd, cwd=None):
#     print(f"\n[CMD] {' '.join(cmd)}")
#     result = subprocess.run(cmd, cwd=cwd)
#     if result.returncode != 0:
#         raise RuntimeError(f"Command failed: {' '.join(cmd)}")


# def check_gpu():
#     print("Checking GPU availability...")
#     print("CUDA available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("GPU:", torch.cuda.get_device_name(0))
#     else:
#         print("WARNING: CUDA not available. GPU optimizations will not work.")


# def ensure_dirs():
#     for d in [MODELS_DIR, RESULTS_DIR]:
#         os.makedirs(d, exist_ok=True)


# # ---------------------------------------------------------------------
# # Pipeline steps
# # ---------------------------------------------------------------------

# def run_profiler():
#     run_cmd([
#         sys.executable,
#         os.path.join(CODE_DIR, "profiling", "pytorch_profiler.py")
#     ])


# def export_onnx():
#     run_cmd([
#         sys.executable,
#         os.path.join(CODE_DIR, "onnx_export", "export_to_onnx.py")
#     ])
#     run_cmd(["ls", "-lh", MODELS_DIR])


# def optimize_onnx():
#     run_cmd([
#         sys.executable,
#         os.path.join(CODE_DIR, "onnx_optim", "optimize_onnx.py")
#     ])
#     run_cmd(["ls", "-lh", MODELS_DIR])


# def benchmark_latency():
#     run_cmd([
#         sys.executable,
#         os.path.join(CODE_DIR, "benchmarking", "benchmark_latency.py")
#     ])


# def compute_wer():
#     run_cmd([
#         sys.executable,
#         os.path.join(CODE_DIR, "benchmarking", "compute_wer.py")
#     ])

# def build_tensorrt_engine():
#     if not shutil.which("trtexec"):
#         print("WARNING: trtexec not found. Skipping TensorRT build.")
#         return

#     run_cmd(["bash", "code/tensorrt/build_engine_fp16.sh"])
#     run_cmd(["bash", "code/tensorrt/build_engine_mixed.sh"])
#     run_cmd(["bash", "code/tensorrt/benchmark_trt.sh"])


# def parse_trt_profiles():
#     profile_path = "models/tensorrt/trt_fp16_profile.json"

#     if not os.path.exists(profile_path):
#         print("[TensorRT] No profile found. Skipping TRT profile parsing.")
#         return

#     run_cmd([
#         sys.executable,
#         "code/tensorrt/parse_trt_profile.py",
#         profile_path,
#         "results/trt_fp16_ops.csv"
#     ])


#     run_cmd([
#         sys.executable,
#         "code/tensorrt/parse_trt_profile.py",
#         "models/tensorrt/trt_mixed_profile.json",
#         "results/trt_mixed_ops.csv",
#     ])

# def compare_results():
#     run_cmd([sys.executable, "code/benchmarking/compare_backends.py"])
#     run_cmd([sys.executable, "code/benchmarking/compare_ops.py"])

# def launch_netron(model_path: str, port: int, label: str):
#     """
#     Launch Netron for a given model artifact on a fixed port.
#     Runs non-blocking so multiple Netron servers can coexist.
#     """
#     if not os.path.exists(model_path):
#         print(f"[Netron] ❌ {label} model not found, skipping.")
#         return

#     if shutil.which("netron") is None:
#         print("[Netron] ❌ Netron is not installed. Run: pip install netron")
#         return

#     print(f"[Netron] Launching {label} graph on port {port}...")

#     subprocess.Popen(
#         [
#             "netron",
#             model_path,
#             "--port",
#             str(port),
#             "--host",
#             "0.0.0.0"
#         ],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL
#     )

# def launch_all_netron():
#     """
#     Launch Netron UIs for PyTorch, ONNX, and TensorRT artifacts.
#     """
#     PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#     pytorch_model = os.path.join(
#         PROJECT_ROOT, "models", "pytorch", "wav2vec2.pt"
#     )
#     onnx_model = os.path.join(
#         PROJECT_ROOT, "models", "onnx", "wav2vec2.onnx"
#     )
#     tensorrt_engine = os.path.join(
#         PROJECT_ROOT, "models", "tensorrt", "wav2vec2_fp16.engine"
#     )

#     launch_netron(pytorch_model, 8081, "PyTorch (TorchScript)")
#     launch_netron(onnx_model, 8082, "ONNX")
#     launch_netron(tensorrt_engine, 8083, "TensorRT")

#     # Small delay to allow servers to start
#     time.sleep(2)

#     print("\n[Netron] Graph viewers available at:")
#     print("  PyTorch   → http://localhost:8081")
#     print("  ONNX      → http://localhost:8082")
#     print("  TensorRT  → http://localhost:8083")

# # --- Helper: select best engine and deploy to Triton ---
# def _read_latency_file(path):
#     try:
#         with open(path, "r") as f:
#             val = float(f.read().strip())
#             return val
#     except Exception:
#         return None

# def select_and_deploy_trt_to_triton(project_root: str):
#     """
#     Look for engines produced by build scripts. Choose best engine (lowest latency)
#     using results/latency_trt_fp16.txt and results/latency_trt_mixed.txt if present,
#     otherwise prefer fp16, then mixed, then fp32.
#     Copy chosen engine into triton/model_repository/wav2vec2/1/model.plan
#     and write config.pbtxt.
#     """
#     print("[Triton] Selecting best TensorRT engine for Triton deployment...")

#     trt_dir = os.path.join(project_root, "models", "tensorrt")
#     candidates = {
#         "fp16": os.path.join(trt_dir, "wav2vec2_fp16.plan"),
#         "mixed": os.path.join(trt_dir, "wav2vec2_mixed.plan"),
#         "fp32": os.path.join(trt_dir, "wav2vec2_fp32.plan"),
#     }

#     # Latency result files (if your benchmark produces these)
#     lat_files = {
#         "fp16": os.path.join(project_root, "results", "latency_trt_fp16.txt"),
#         "mixed": os.path.join(project_root, "results", "latency_trt_mixed.txt"),
#         "fp32": os.path.join(project_root, "results", "latency_trt_fp32.txt"),
#     }

#     scores = {}
#     for k, p in candidates.items():
#         if os.path.exists(p):
#             # default score: very high (bad) if no latency info
#             lat = _read_latency_file(lat_files.get(k))
#             scores[k] = lat if lat is not None else float("inf")
#         else:
#             scores[k] = None

#     # Filter only available engines
#     available = {k: v for k, v in candidates.items() if scores.get(k) is not None}

#     if not available:
#         print("[Triton] No TRT engines found; skipping Triton deployment.")
#         return

#     # Choose min-latency engine (latencies with inf treat as unknown)
#     # Prefer lower latency; if all unknown, prefer fp16 -> mixed -> fp32
#     chosen = None
#     # If any have real latency (not inf), pick min
#     real_latencies = {k: s for k, s in scores.items() if s not in (None, float("inf"))}
#     if real_latencies:
#         chosen = min(real_latencies, key=real_latencies.get)
#     else:
#         # prefer order fp16, mixed, fp32
#         for pref in ("fp16", "mixed", "fp32"):
#             if scores.get(pref) is not None:
#                 chosen = pref
#                 break

#     if chosen is None:
#         print("[Triton] Unable to choose engine. Aborting deployment.")
#         return

#     chosen_path = candidates[chosen]
#     print(f"[Triton] Chosen engine: {chosen} -> {chosen_path}")

#     # Create triton model repository folders
#     triton_model_dir = os.path.join(project_root, "triton", "model_repository", "wav2vec2")
#     version_dir = os.path.join(triton_model_dir, "1")
#     os.makedirs(version_dir, exist_ok=True)

#     # Copy chosen engine as model.plan
#     dst_engine = os.path.join(version_dir, "model.plan")
#     shutil.copyfile(chosen_path, dst_engine)
#     print(f"[Triton] Copied engine to {dst_engine}")

#     # Write config.pbtxt
#     cfg = f"""
#             name: "wav2vec2"
#             backend: "tensorrt"
#             max_batch_size: 8

#             input [
#             {{
#                 name: "input_values"
#                 data_type: TYPE_FP32
#                 dims: [-1, -1]
#             }}
#             ]

#             output [
#             {{
#                 name: "logits"
#                 data_type: TYPE_FP32
#                 dims: [-1, -1]
#             }}
#             ]

#             instance_group [
#             {{
#                 kind: KIND_GPU
#                 count: 1
#             }}
#             ]
#         """
#     cfg_path = os.path.join(triton_model_dir, "config.pbtxt")
#     with open(cfg_path, "w") as f:
#         f.write(cfg.strip())
#     print(f"[Triton] Wrote Triton config at {cfg_path}")

#     print("[Triton] Model deployed to model_repository/wav2vec2. You can now start Triton Server.")

#     run_cmd([
#         "bash",
#         "tritonserver",
#         "--model-repository=triton/model_repository",
#         "--strict-model-config=false"
#     ])

# # ---------------------------------------------------------------------
# # Main entry point
# # ---------------------------------------------------------------------

# def main():
#     print("=" * 60)
#     print("AI4Bharat ASR Optimization – Local Pipeline")
#     print("=" * 60)

#     ensure_dirs()
#     check_gpu()

#     print("\n[STEP 1] PyTorch Profiling")
#     run_profiler()

#     print("\n[STEP 2] Export ONNX")
#     export_onnx()

#     print("\n[STEP 3] Optimize ONNX (INT8 bypass)")
#     optimize_onnx()

#     print("\n[STEP 4] Benchmark ONNX Runtime")
#     benchmark_latency()

#     print("\n[STEP 5] Compute WER")
#     compute_wer()

#     print("\n[STEP 6] TensorRT Engine Build")
#     build_tensorrt_engine()

#     print("\n[STEP 6.5] Deploy best TRT engine to Triton model_repository")
#     select_and_deploy_trt_to_triton(PROJECT_ROOT)

#     print("\n[STEP 7] Parse TRT Profiles")
#     parse_trt_profiles()

#     print("\n[STEP 8] Compare Results")
#     compare_results()

#     print("\n[INFO] Launching Netron graph visualizers...")
#     launch_all_netron()

#     print("\nPipeline completed successfully.")

# if __name__ == "__main__":
#     main()

import os
import sys
import subprocess
import torch
import shutil
import time
from pathlib import Path

# PROJECT layout
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# helpers
def run_cmd(cmd, cwd=None, env=None, show_cmd=True):
    if show_cmd:
        print(f"\n[CMD] {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")

def check_gpu():
    print("Checking GPU availability...")
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

# pipeline steps
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
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "benchmark_latency.py")])

def compute_wer():
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "compute_wer.py")])

def build_tensorrt_engine():
    build_dir = CODE_DIR / "tensorrt"
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        print("WARNING: trtexec not found on PATH. Skipping TensorRT engine build.")
        return
    # call the build scripts if present
    fp16 = build_dir / "build_engine_fp16.sh"
    mixed = build_dir / "build_engine_mixed.sh"
    bench = build_dir / "benchmark_trt.sh"
    if fp16.exists():
        run_cmd(["bash", str(fp16)], cwd=str(build_dir))
    if mixed.exists():
        run_cmd(["bash", str(mixed)], cwd=str(build_dir))
    if bench.exists():
        run_cmd(["bash", str(bench)], cwd=str(build_dir))

def parse_trt_profiles():
    # two profiles we might have
    p1 = MODELS_DIR / "tensorrt" / "trt_fp16_profile.json"
    p2 = MODELS_DIR / "tensorrt" / "trt_mixed_profile.json"
    parser = CODE_DIR / "tensorrt" / "parse_trt_profile.py"
    if p1.exists():
        run_cmd([sys.executable, str(parser), str(p1), str(RESULTS_DIR / "trt_fp16_ops.csv")])
    else:
        print("[TensorRT] No fp16 profile found; skipping.")
    if p2.exists():
        run_cmd([sys.executable, str(parser), str(p2), str(RESULTS_DIR / "trt_mixed_ops.csv")])
    else:
        print("[TensorRT] No mixed profile found; skipping.")

def compare_results():
    # compare_backends will expect latency files (latency_pytorch.txt etc)
    # if missing, it may fail; call it and let it raise so user knows to produce latency files
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "compare_backends.py")])
    run_cmd([sys.executable, str(CODE_DIR / "benchmarking" / "compare_ops.py")])

def _choose_trt_engine(project_root: Path):
    # accept .plan or .engine and several file names
    trt_dir = project_root / "models" / "tensorrt"
    candidates = {
        "fp16": [trt_dir / "wav2vec2_fp16.plan", trt_dir / "wav2vec2_fp16.engine", trt_dir / "wav2vec2_fp16.trt"],
        "mixed": [trt_dir / "wav2vec2_mixed.plan", trt_dir / "wav2vec2_mixed.engine"],
        "fp32": [trt_dir / "wav2vec2_fp32.plan", trt_dir / "wav2vec2_fp32.engine"],
    }
    found = {}
    for k, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[k] = p
                break
    return found

def select_and_deploy_trt_to_triton(project_root: Path):
    print("[Triton] Attempting to select a TRT engine for Triton model_repository...")
    found = _choose_trt_engine(project_root)
    if not found:
        print("[Triton] No TensorRT engine files found in models/tensorrt/. Skipping Triton deployment.")
        return

    # read latencies if present
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
            scores[k] = float("inf")  # unknown → large

    # choose min latency
    chosen = min(scores, key=scores.get)
    engine_path = found[chosen]
    print(f"[Triton] Chosen engine: {chosen} -> {engine_path}")

    # copy to triton model repository
    triton_root = project_root / "triton" / "model_repository" / "wav2vec2"
    version_dir = triton_root / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    dst_engine = version_dir / "model.plan"
    shutil.copyfile(str(engine_path), str(dst_engine))
    print(f"[Triton] Copied engine to {dst_engine}")

    # ensure config.pbtxt (minimal)
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
    print(f"[Triton] Wrote config to {cfg_path}")

    triton_exe = shutil.which("tritonserver")
    if triton_exe is None:
        print("[Triton] tritonserver not found on PATH. To run Triton manually:")
        print(f"  tritonserver --model-repository={project_root / 'triton' / 'model_repository'} --strict-model-config=false")
        return

    # everything ready — start triton server (user can stop with ctrl-c)
    print("[Triton] Starting tritonserver (this will block until terminated)...")
    run_cmd([triton_exe, f"--model-repository={project_root/'triton'/'model_repository'}", "--strict-model-config=false"])

def _choose_netron_candidate(paths):
    for p in paths:
        if p.exists():
            return str(p)
    return None

def launch_netron(model_path: str, port: int, label: str):
    if not shutil.which("netron"):
        print("[Netron] netron not installed (pip install netron). Skipping Netron for", label)
        return
    if not Path(model_path).exists():
        print(f"[Netron] Model not found for {label}: {model_path}; skipping.")
        return
    print(f"[Netron] Launching {label} on port {port}...")
    subprocess.Popen(["netron", model_path, "--port", str(port), "--host", "0.0.0.0"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def launch_all_netron():
    # check common candidate paths
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

    # allow servers to come up slightly
    time.sleep(2)
    print("\n[Netron] Graph viewers (if launched):")
    if p_torch:
        print("  PyTorch → http://localhost:8081")
    if p_onnx:
        print("  ONNX    → http://localhost:8082")
    if p_trt:
        print("  TRT     → http://localhost:8083")

def main():
    print("=" * 60)
    print("AI4Bharat ASR Optimization – Local Pipeline")
    print("=" * 60)
    ensure_dirs()
    check_gpu()

    print("\n[STEP 1] PyTorch Profiling")
    try:
        run_profiler()
    except Exception as e:
        print("[STEP 1] failed:", e)
        # continue, perhaps ONNX is available

    print("\n[STEP 2] Export ONNX")
    try:
        export_onnx()
    except Exception as e:
        print("[STEP 2] ONNX export failed:", e)

    print("\n[STEP 3] Optimize ONNX")
    try:
        optimize_onnx()
    except Exception as e:
        print("[STEP 3] ONNX optimization failed:", e)

    print("\n[STEP 4] Benchmark ONNX Runtime")
    try:
        benchmark_latency()
    except Exception as e:
        print("[STEP 4] Benchmark failed:", e)

    print("\n[STEP 5] Compute WER")
    try:
        compute_wer()
    except Exception as e:
        print("[STEP 5] WER computation failed:", e)

    print("\n[STEP 6] TensorRT Engine Build")
    try:
        build_tensorrt_engine()
    except Exception as e:
        print("[STEP 6] TRT build failed:", e)

    print("\n[STEP 6.5] Deploy best TRT to Triton (if available)")
    try:
        select_and_deploy_trt_to_triton(PROJECT_ROOT)
    except Exception as e:
        print("[STEP 6.5] Triton deployment failed:", e)

    print("\n[STEP 7] Parse TRT Profiles")
    try:
        parse_trt_profiles()
    except Exception as e:
        print("[STEP 7] TRT profile parsing failed:", e)

    print("\n[STEP 8] Compare Results")
    try:
        compare_results()
    except Exception as e:
        print("[STEP 8] Compare failed:", e)

    print("\n[INFO] Launching Netron graph viewers (if netron installed)...")
    launch_all_netron()

    print("\nPipeline finished (some steps may have been skipped or failed).")

if __name__ == "__main__":
    main()