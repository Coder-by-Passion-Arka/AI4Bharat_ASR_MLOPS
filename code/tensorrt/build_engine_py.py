#!/usr/bin/env python3
"""
Build TensorRT engine(s) from ONNX using TensorRT Python API.
Production-ready, robust, and fully logged.
"""

import argparse
import os
import time
import json
import logging
from pathlib import Path

import onnx
import tensorrt as trt

from code.utils.logger import get_logger

# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------
logger = get_logger(name="build_engine_py", level=logging.DEBUG)

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def get_onnx_input_name(onnx_path: Path) -> str:
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializer_names = {i.name for i in graph.initializer}
    for inp in graph.input:
        if inp.name not in initializer_names:
            return inp.name
    return graph.input[0].name


def validate_shapes(min_s, opt_s, max_s):
    for i, (a, b, c) in enumerate(zip(min_s, opt_s, max_s)):
        if not (a <= b <= c):
            raise ValueError(
                f"Invalid optimization profile dimension {i}: "
                f"min={a}, opt={b}, max={c}"
            )


# ------------------------------------------------------------------
# Engine builder
# ------------------------------------------------------------------
def build_engine(
    onnx_path: Path,
    out_path: Path,
    precision: str,
    input_name: str | None,
    min_shape,
    opt_shape,
    max_shape,
    workspace_gb: int,
    verbose: bool = False,
):
    TRT_LOGGER = trt.Logger(
        trt.Logger.VERBOSE if verbose else trt.Logger.ERROR
    )

    onnx_path = onnx_path.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    validate_shapes(min_shape, opt_shape, max_shape)

    if input_name is None:
        input_name = get_onnx_input_name(onnx_path)
        logger.info(f"Guessed ONNX input name → {input_name}")

    logger.info(f"TensorRT version: {trt.__version__}")
    logger.info(f"Precision mode: {precision.upper()}")
    logger.info(f"Workspace: {workspace_gb} GB")
    logger.info(f"Shapes: min={min_shape}, opt={opt_shape}, max={max_shape}")

    # IMPORTANT: external-data resolution
    old_cwd = os.getcwd()
    os.chdir(str(onnx_path.parent))

    try:
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        logger.info(f"Parsing ONNX → {onnx_path.name}")
        with open(onnx_path.name, "rb") as f:
            if not parser.parse(f.read()):
                logger.error("ONNX parse failed")
                for i in range(parser.num_errors):
                    logger.error(parser.get_error(i))
                raise RuntimeError("TensorRT ONNX parsing failed")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * (1 << 30)),
        )

        if precision in ("fp16", "mixed"):
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 enabled")
            else:
                logger.warning("FP16 requested but platform lacks fast FP16")

        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        logger.info("Building TensorRT engine …")
        t0 = time.time()
        engine = builder.build_serialized_network(network, config)
        t1 = time.time()

        if engine is None:
            raise RuntimeError("TensorRT returned None engine")

        with open(out_path, "wb") as f:
            f.write(engine)

        logger.info(f"Engine built in {t1 - t0:.1f}s → {out_path}")

        # Metadata for examiners
        meta = {
            "onnx": str(onnx_path),
            "engine": str(out_path),
            "precision": precision,
            "input": input_name,
            "shapes": {
                "min": min_shape,
                "opt": opt_shape,
                "max": max_shape,
            },
            "tensorrt_version": trt.__version__,
            "build_time_sec": round(t1 - t0, 2),
        }

        meta_path = out_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Engine metadata saved → {meta_path}")
        return out_path

    finally:
        os.chdir(old_cwd)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("TensorRT Engine Builder (Python API)")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16", "mixed"], default="fp16")
    ap.add_argument("--input-name", default=None)
    ap.add_argument("--min", nargs=2, type=int, default=(1, 16000))
    ap.add_argument("--opt", nargs=2, type=int, default=(4, 64000))
    ap.add_argument("--max", nargs=2, type=int, default=(8, 128000))
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_name = f"{Path(args.onnx).stem}_trt_{args.precision}.plan"
    out_path = Path(args.out_dir) / out_name

    build_engine(
        Path(args.onnx),
        out_path,
        args.precision,
        args.input_name,
        tuple(args.min),
        tuple(args.opt),
        tuple(args.max),
        args.workspace_gb,
        args.verbose,
    )


if __name__ == "__main__":
    main()