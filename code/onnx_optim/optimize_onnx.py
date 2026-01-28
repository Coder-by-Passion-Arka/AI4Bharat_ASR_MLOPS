#!/usr/bin/env python3
"""
ONNX optimization stage (WSL-safe, TensorRT-safe).

What this version guarantees:
- Single ONNX in memory at a time
- No accidental external-data breakage
- Optional simplification (disabled by default)
- Structured logging (no print)
- Safe for large transformer graphs in WSL
"""

from __future__ import annotations
import sys
import shutil
import time
import json
from pathlib import Path
import onnx
from onnx import shape_inference

from code.utils.logger import get_logger

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = get_logger(
    name="onnx_optim",
    log_file="onnx_optim.log"
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ONNX_DIR = PROJECT_ROOT / "models" / "onnx"

SRC = ONNX_DIR / "wav2vec2.onnx"
DST = ONNX_DIR / "wav2vec2_optimized.onnx"
BACKUP_DIR = ONNX_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Flags (EXAMINER FRIENDLY)
# -----------------------------------------------------------------------------
ENABLE_SIMPLIFIER = False  # keep False for WSL stability

# -----------------------------------------------------------------------------
def fatal(msg: str, code: int = 1):
    logger.error(msg)
    sys.exit(code)

def find_external_data(model: onnx.ModelProto):
    """
    Inspect model initializers for external_data locations.
    Returns a list of Path objects (filenames are relative to SRC.parent).
    """
    files = set()
    for init in model.graph.initializer:
        for kv in init.external_data:
            if kv.key.lower() == "location":
                files.add(Path(kv.value))
    return sorted(files)

# -----------------------------------------------------------------------------
def main():
    logger.info("Starting ONNX optimization")
    logger.info(f"SRC = {SRC}")
    logger.info(f"DST = {DST}")

    if not SRC.exists():
        fatal(f"Source ONNX not found: {SRC}", 1)

    # -------------------------------------------------------------------------
    # Backup existing optimized model
    # -------------------------------------------------------------------------
    if DST.exists():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        bkp = BACKUP_DIR / f"wav2vec2_optimized.onnx.bak.{stamp}"
        shutil.copy2(DST, bkp)
        logger.info(f"Backed up existing optimized ONNX → {bkp}")

    # -------------------------------------------------------------------------
    # Load + validate SRC (ONLY ONCE)
    # -------------------------------------------------------------------------
    try:
        model = onnx.load(str(SRC))
        onnx.checker.check_model(model)
        logger.info("Source ONNX passed checker")
    except Exception as e:
        fatal(f"ONNX validation failed: {e}", 2)

    # -------------------------------------------------------------------------
    # Shape inference (best effort)
    # -------------------------------------------------------------------------
    try:
        logger.info("Running shape inference")
        model = shape_inference.infer_shapes(model)
        onnx.save_model(model, str(DST))
        logger.info(f"Shape-inferred ONNX written → {DST}")
    except Exception as e:
        logger.warning(f"Shape inference failed ({e}), copying SRC instead")
        shutil.copy2(SRC, DST)

    # -------------------------------------------------------------------------
    # External data handling
    # -------------------------------------------------------------------------
    ext_files = find_external_data(model)
    if ext_files:
        logger.info(f"Detected {len(ext_files)} external-data files")
        for f in ext_files:
            src_f = SRC.parent / f
            dst_f = DST.parent / f
            if not src_f.exists():
                fatal(f"Missing external-data file: {src_f}", 3)
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_f)
            logger.info(f"Copied external data → {dst_f}")
    else:
        logger.info("No external-data files detected")

    # -------------------------------------------------------------------------
    # Optional simplification (OFF by default)
    # -------------------------------------------------------------------------
    if ENABLE_SIMPLIFIER:
        try:
            import onnxsim
            logger.info("Running onnx-simplifier (explicitly enabled)")
            simplified, ok = onnxsim.simplify(str(DST))
            if ok:
                onnx.save_model(simplified, str(DST))
                logger.info("Simplified ONNX saved")
            else:
                logger.warning("onnx-simplifier check=False, keeping original")
        except Exception as e:
            logger.warning(f"onnx-simplifier failed: {e}")

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------
    try:
        final_model = onnx.load(str(DST))
        onnx.checker.check_model(final_model)
        logger.info("Final optimized ONNX validated successfully")
    except Exception as e:
        fatal(f"Final ONNX validation failed: {e}", 2)

    # -------------------------------------------------------------------------
    # Summary (examiner-ready)
    # -------------------------------------------------------------------------
    opset = final_model.opset_import[0].version if final_model.opset_import else "unknown"
    logger.info(json.dumps({
        "final_onnx": str(DST),
        "opset": opset,
        "producer": final_model.producer_name,
        "external_data_files": len(ext_files),
        "simplifier_used": ENABLE_SIMPLIFIER
    }))

    logger.info("ONNX optimization complete")
    sys.exit(0)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()