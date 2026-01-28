#!/usr/bin/env python3
"""
Compute Word Error Rate (WER) on Common Voice dataset.

This script:
- Loads Common Voice (legacy-datasets)
- Uses Wav2Vec2Processor for preprocessing & decoding
- Runs inference on a small subset
- Computes and saves WER

NOTE:
This script evaluates accuracy, not performance.
"""

import os
import csv
import torch
import logging
import evaluate
# import torchaudio
import soundfile as sf
import torchaudio.functional as F
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "common_voice", "hi")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
TSV_PATH = os.path.join(DATA_DIR, "validated.tsv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

from code.utils.logger import get_logger

logger = get_logger(level=logging.DEBUG)

logger.info("Computing Word Entity Recognition from Common Voice Hindi Dataset")

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "ai4bharat/indicwav2vec-hindi"
LANGUAGE = "hi"
TARGET_SR = 16000
MAX_SAMPLES = 20
DEVICE = "cpu"  # correctness-only

# -----------------------------
# Load model
# -----------------------------
print("[WER] Loading model and processor...")
model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

# -----------------------------
# Load TSV metadata
# -----------------------------
print("[WER] Loading Common Voice Hindi metadata...")
wer_metric = evaluate.load("wer")
samples = []

with open(TSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["sentence"].strip():
            samples.append(row)
        if len(samples) >= MAX_SAMPLES:
            break

print(f"[WER] Loaded {len(samples)} samples")

# -----------------------------
# Inference loop
# -----------------------------

references, predictions = [], []

print(f"[WER] Running inference on {MAX_SAMPLES} samples...")

for sample in tqdm(samples):
    audio_path = os.path.join(CLIPS_DIR, sample["path"])

    waveform, sr = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.ndim > 1:  # stereo → mono
        waveform = waveform.mean(dim=1)

    if sr != TARGET_SR:
        waveform = F.resample(waveform, sr, TARGET_SR)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)

    transcription = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )[0]

    references.append(sample["sentence"].lower())
    predictions.append(transcription.lower())

# -----------------------------
# Compute WER
# -----------------------------

wer_value = wer_metric.compute(
    predictions=predictions,
    references=references
)

print(f"[WER] Final WER (Hindi): {wer_value:.6f}")

out_path = os.path.join(RESULTS_DIR, f"WER_{LANGUAGE}.txt")
with open(out_path, "w") as f:
    f.write(f"WER: {wer_value:.6f}\n")
    f.write(f"Samples evaluated: {len(predictions)}\n")

print(f"[WER] Results saved to {out_path}")