#!/bin/bash
# Runs Triton Model Analyzer

model-analyzer profile \
  --model-repository ./model_repository \
  --profile-models wav2vec2 \
  --triton-launch-mode local
