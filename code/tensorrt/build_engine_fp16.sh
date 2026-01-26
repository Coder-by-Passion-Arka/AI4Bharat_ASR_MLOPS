#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENGINE_DIR="${PROJECT_ROOT}/models/tensorrt"
ONNX_PATH="${PROJECT_ROOT}/models/onnx/wav2vec2.onnx"
ENGINE_NAME="wav2vec2_trt_fp16.plan"
ENGINE_PATH="${ENGINE_DIR}/${ENGINE_NAME}"

mkdir -p "${ENGINE_DIR}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "[ERROR] trtexec not found in PATH."
  exit 2
fi

echo "[FP16] Building TensorRT FP16 engine..."
trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --dumpProfile \
  --exportProfile="${ENGINE_DIR}/trt_fp16_profile.json" \
  --minShapes="input_values:1x16000" \
  --optShapes="input_values:4x64000" \
  --maxShapes="input_values:8x128000" \
  --buildOnly \
  || { echo "[FP16] trtexec failed."; exit 3; }

echo "[FP16] TensorRT engine written to: ${ENGINE_PATH}"
