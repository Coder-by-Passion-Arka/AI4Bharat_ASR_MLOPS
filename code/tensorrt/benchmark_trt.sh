#!/usr/bin/env bash

INPUT_SHAPE="input_values:1x16000"
WARMUP=10
RUNS=50

echo "==== TensorRT FP16 ===="
/usr/local/cuda/bin/trtexec \
  --loadEngine=models/tensorrt/wav2vec2_trt_fp16.engine \
  --shapes=${INPUT_SHAPE} \
  --warmUp=${WARMUP} \
  --iterations=${RUNS}
echo "$AVG_LATENCY_MS" > results/latency_trt_fp16.txt

echo ""
echo "==== TensorRT Mixed Precision ===="
/usr/local/cuda/bin/trtexec \
  --loadEngine=models/tensorrt/wav2vec2_trt_mixed.engine \
  --shapes=${INPUT_SHAPE} \
  --warmUp=${WARMUP} \
  --iterations=${RUNS}
echo "$AVG_LATENCY_MS" > results/latency_trt_mixed.txt
