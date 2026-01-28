#!/usr/bin/env python3
"""
Triton Server Verification & Performance Check

Features:
- Server & model readiness check
- Automatic input/output discovery
- Warmup + latency benchmarking
- Percentiles (p50/p90/p99)
- Output sanity validation
- Writes both .txt and .json latency artifacts
"""

import time
import json
import numpy as np
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def connect():
    logger.info("[Triton] Connecting to Triton server...")
    try:
        return httpclient.InferenceServerClient(
            url=TRITON_URL,
            verbose=True,
            # timeout=5.0
        )
    except Exception as e:
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

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    logger.info("========== Triton Inference Validation ==========")

    client = connect()
    check_health(client)
    check_model(client)

    INPUT_NAME, OUTPUT_NAME = discover_io(client)

    audio = build_dummy_audio()
    inputs = [
        httpclient.InferInput(INPUT_NAME, audio.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(audio)

    outputs = [
        httpclient.InferRequestedOutput(OUTPUT_NAME)
    ]

    # Warmup
    logger.info("[Triton] Warmup runs...")
    for _ in range(N_WARMUP):
        client.infer(MODEL_NAME, inputs, outputs=outputs)

    # Benchmark
    logger.info("[Triton] Benchmarking inference latency...")
    latencies = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = client.infer(MODEL_NAME, inputs, outputs=outputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Sanity check
        out = result.as_numpy(OUTPUT_NAME)
        if out is None or not np.isfinite(out).all():
            raise RuntimeError("Invalid Triton output detected")

    avg = float(np.mean(latencies))
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p99 = float(np.percentile(latencies, 99))

    logger.info(f"[Triton] Avg latency : {avg:.2f} ms")
    logger.info(f"[Triton] P50 latency : {p50:.2f} ms")
    logger.info(f"[Triton] P90 latency : {p90:.2f} ms")
    logger.info(f"[Triton] P99 latency : {p99:.2f} ms")

    stats = {
        "backend": "triton",
        "avg_latency_ms": avg, 
        "p50_ms": p50,
        "p90_ms": p90,
        "p99_ms": p99,
        "runs": N_RUNS,
        "unit": "milliseconds",
    }


    (RESULTS_DIR / "latency_triton.json").write_text(
        json.dumps(stats, indent=2)
    )
    (RESULTS_DIR / "latency_triton.txt").write_text(f"{avg:.6f} # milliseconds")

    logger.info("[Triton] Results saved to results/")
    logger.info("========== Triton validation complete ==========")

if __name__ == "__main__":
    main()