#!/usr/bin/env bash
#
# Build a TensorRT mixed-precision engine using trtexec.
# Saves engine to models/tensorrt/wav2vec2_trt_mixed.plan
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENGINE_DIR="${PROJECT_ROOT}/models/tensorrt"
ONNX_PATH="${PROJECT_ROOT}/models/onnx/wav2vec2.onnx"
ENGINE_NAME="wav2vec2_trt_mixed.plan"
ENGINE_PATH="${ENGINE_DIR}/${ENGINE_NAME}"

mkdir -p "${ENGINE_DIR}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "[ERROR] trtexec not found in PATH. Make sure TensorRT is installed and trtexec is in PATH."
  exit 2
fi

echo "[Mixed] Building TensorRT mixed-precision engine..."
# Use conservative shapes: batch x time (time = 16000 is 1s). Adjust opt/max as needed.
trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --dumpProfile \
  --exportProfile="${ENGINE_DIR}/trt_mixed_profile.json" \
  --minShapes="input_values:1x16000" \
  --optShapes="input_values:4x64000" \
  --maxShapes="input_values:8x128000" \
  --buildOnly \
  || {
    echo "[Mixed] trtexec failed, aborting."
    exit 3
  }

echo "[Mixed] TensorRT mixed-precision engine written to: ${ENGINE_PATH}"
