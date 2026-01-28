# #!/usr/bin/env python3
# """
# Triton Server Verification & Performance Check

# Features:
# - Server & model readiness check
# - Automatic input/output discovery
# - Warmup + latency benchmarking
# - Percentiles (p50/p90/p99)
# - Output sanity validation
# - Writes both .txt and .json latency artifacts
# """

# import time
# import json
# import numpy as np
# from pathlib import Path

# import tritonclient.http as httpclient
# from tritonclient.utils import InferenceServerException

# from code.utils.logger import get_logger

# logger = get_logger(name="triton_check", level=10)

# # ---------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------
# TRITON_URL = "localhost:8000"
# MODEL_NAME = "wav2vec2"

# N_WARMUP = 5
# N_RUNS = 20

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR = PROJECT_ROOT / "results"
# RESULTS_DIR.mkdir(exist_ok=True)

# # ---------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------
# def connect():
#     logger.info("[Triton] Connecting to Triton server...")
#     try:
#         return httpclient.InferenceServerClient(
#             url=TRITON_URL,
#             verbose=True,
#             # timeout=5.0
#         )
#     except Exception as e:
#         logger.exception("Failed to connect to Triton server")
#         raise SystemExit(1)

# def check_health(client):
#     if not client.is_server_ready():
#         raise RuntimeError("Triton server not ready")
#     logger.info("[Triton] Server is READY")

# def check_model(client):
#     if not client.is_model_ready(MODEL_NAME):
#         raise RuntimeError(f"Model '{MODEL_NAME}' not ready")
#     logger.info(f"[Triton] Model '{MODEL_NAME}' is READY")

# def discover_io(client):
#     meta = client.get_model_metadata(MODEL_NAME)
#     inp = meta["inputs"][0]["name"]
#     out = meta["outputs"][0]["name"]
#     logger.info(f"[Triton] Discovered input  → {inp}")
#     logger.info(f"[Triton] Discovered output → {out}")
#     return inp, out

# def build_dummy_audio(batch=1, length=16000):
#     return np.random.randn(batch, length).astype(np.float32)

# # ---------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------
# def main():
#     logger.info("========== Triton Inference Validation ==========")

#     client = connect()
#     check_health(client)
#     check_model(client)

#     INPUT_NAME, OUTPUT_NAME = discover_io(client)

#     audio = build_dummy_audio()
#     inputs = [
#         httpclient.InferInput(INPUT_NAME, audio.shape, "FP32")
#     ]
#     inputs[0].set_data_from_numpy(audio)

#     outputs = [
#         httpclient.InferRequestedOutput(OUTPUT_NAME)
#     ]

#     # Warmup
#     logger.info("[Triton] Warmup runs...")
#     for _ in range(N_WARMUP):
#         client.infer(MODEL_NAME, inputs, outputs=outputs)

#     # Benchmark
#     logger.info("[Triton] Benchmarking inference latency...")
#     latencies = []

#     for _ in range(N_RUNS):
#         t0 = time.perf_counter()
#         result = client.infer(MODEL_NAME, inputs, outputs=outputs)
#         t1 = time.perf_counter()
#         latencies.append((t1 - t0) * 1000.0)

#         # Sanity check
#         out = result.as_numpy(OUTPUT_NAME)
#         if out is None or not np.isfinite(out).all():
#             raise RuntimeError("Invalid Triton output detected")

#     avg = float(np.mean(latencies))
#     p50 = float(np.percentile(latencies, 50))
#     p90 = float(np.percentile(latencies, 90))
#     p99 = float(np.percentile(latencies, 99))

#     logger.info(f"[Triton] Avg latency : {avg:.2f} ms")
#     logger.info(f"[Triton] P50 latency : {p50:.2f} ms")
#     logger.info(f"[Triton] P90 latency : {p90:.2f} ms")
#     logger.info(f"[Triton] P99 latency : {p99:.2f} ms")

#     stats = {
#         "backend": "triton",
#         "avg_latency_ms": avg, 
#         "p50_ms": p50,
#         "p90_ms": p90,
#         "p99_ms": p99,
#         "runs": N_RUNS,
#         "unit": "milliseconds",
#     }


#     (RESULTS_DIR / "latency_triton.json").write_text(
#         json.dumps(stats, indent=2)
#     )
#     (RESULTS_DIR / "latency_triton.txt").write_text(f"{avg:.6f} # milliseconds")

#     logger.info("[Triton] Results saved to results/")
#     logger.info("========== Triton validation complete ==========")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Triton Server Verification & Performance Check (ENHANCED)

Added:
- Automatic Triton availability & load verification
- Batch-size stress testing
- QPS benchmarking
- Graceful Triton shutdown (optional)
- Safe exit if Triton is not running
"""

import time
import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import requests
from pathlib import Path

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from code.utils.logger import get_logger

logger = get_logger(name="triton_check", level=10)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TRITON_URL = "localhost:8000"
MODEL_NAME = "wav2vec2"

N_WARMUP = 5
N_RUNS = 20

# Stress-test batches (simple example)
BATCH_SIZES = [1, 2, 4, 8]

# Graceful shutdown toggle
ENABLE_SHUTDOWN = False  # set True if you want auto shutdown

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------
def triton_reachable() -> bool:
    try:
        r = requests.get(f"http://{TRITON_URL}/v2/health/ready", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def connect():
    logger.info("[Triton] Connecting to Triton server...")
    try:
        return httpclient.InferenceServerClient(
            url=TRITON_URL,
            verbose=False,
        )
    except Exception:
        logger.exception("Failed to connect to Triton server")
        raise SystemExit(1)

def check_health(client):
    if not client.is_server_ready():
        raise RuntimeError("Triton server not ready")
    logger.info("[Triton] Server is READY")

def check_model(client):
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' not ready")
    logger.info(f"[Triton] Model '{MODEL_NAME}' is READY")

def discover_io(client):
    meta = client.get_model_metadata(MODEL_NAME)
    inp = meta["inputs"][0]["name"]
    out = meta["outputs"][0]["name"]
    logger.info(f"[Triton] Discovered input  → {inp}")
    logger.info(f"[Triton] Discovered output → {out}")
    return inp, out

def build_dummy_audio(batch=1, length=16000):
    return np.random.randn(batch, length).astype(np.float32)

def plot_stress_results(results, out_path: Path):
    batch_sizes = [r["batch_size"] for r in results]
    avg_latency = [r["avg_ms"] for r in results]
    qps = [r["qps"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Avg Latency (ms)", color=color1)
    ax1.plot(batch_sizes, avg_latency, marker="o", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("QPS", color=color2)
    ax2.plot(batch_sizes, qps, marker="s", linestyle="--", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Triton Stress Test: Latency & Throughput")
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(out_path, dpi=150)
    plt.close()

    logger.info(f"[Triton] Stress-test plot saved → {out_path}")

# ---------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------
def benchmark_batch(client, input_name, output_name, batch_size):
    audio = build_dummy_audio(batch=batch_size)

    inputs = [httpclient.InferInput(input_name, audio.shape, "FP32")]
    inputs[0].set_data_from_numpy(audio)
    outputs = [httpclient.InferRequestedOutput(output_name)]

    # Warmup
    for _ in range(N_WARMUP):
        client.infer(MODEL_NAME, inputs, outputs=outputs)

    latencies = []
    t_start = time.perf_counter()

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = client.infer(MODEL_NAME, inputs, outputs=outputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        out = result.as_numpy(output_name)
        if out is None or not np.isfinite(out).all():
            raise RuntimeError("Invalid Triton output detected")

    total_time = time.perf_counter() - t_start
    qps = (N_RUNS * batch_size) / total_time

    return {
        "batch_size": batch_size,
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "qps": float(qps),
    }

# ---------------------------------------------------------------------
# Optional graceful shutdown
# ---------------------------------------------------------------------
def shutdown_triton():
    logger.info("[Triton] Attempting graceful shutdown...")
    try:
        subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=nvcr.io/nvidia/tritonserver"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            ["docker", "stop", "$(docker ps -q --filter ancestor=nvcr.io/nvidia/tritonserver)"],
            shell=True,
        )
        logger.info("[Triton] Triton server stopped")
    except Exception:
        logger.warning("[Triton] Failed to stop Triton (may not be running)")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("========== Triton Inference Validation ==========")

    # 🔒 Safety: don’t crash pipeline if Triton isn’t running
    if not triton_reachable():
        logger.warning("[Triton] Server not reachable — skipping Triton validation")
        return

    client = connect()
    check_health(client)
    check_model(client)

    input_name, output_name = discover_io(client)

    all_results = []

    logger.info("[Triton] Starting batch-size stress test + QPS benchmarking")

    for bs in BATCH_SIZES:
        logger.info(f"[Triton] Testing batch size = {bs}")
        stats = benchmark_batch(client, input_name, output_name, bs)
        all_results.append(stats)

        logger.info(
            f"[Triton][BS={bs}] "
            f"Avg={stats['avg_ms']:.2f} ms | "
            f"P90={stats['p90_ms']:.2f} ms | "
            f"QPS={stats['qps']:.2f}"
        )

    # Save artifacts
    out_json = RESULTS_DIR / "latency_triton_stress.json"
    out_txt = RESULTS_DIR / "latency_triton_stress.txt"

    out_json.write_text(json.dumps(all_results, indent=2))
    out_txt.write_text(
        "\n".join(
            f"BS={r['batch_size']} | Avg={r['avg_ms']:.3f} ms | QPS={r['qps']:.2f}"
            for r in all_results
        )
    )

    logger.info("[Triton] Stress-test results saved to results/")
	
    plot_stress_results(
        all_results,
        RESULTS_DIR / "triton_stress_test.png"
    )

    if ENABLE_SHUTDOWN:
        shutdown_triton()

    logger.info("========== Triton validation complete ==========")

if __name__ == "__main__":
    main()