# AI4Bharat ASR Model Optimization

**High-Performance Inference Pipeline for Wav2Vec2-based Hindi Speech Recognition**

This project optimizes the [AI4Bharat IndicWav2Vec Hindi](https://huggingface.co/ai4bharat/indicwav2vec-hindi) Automatic Speech Recognition (ASR) model for production deployment using NVIDIA's inference stack.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Documentation](#documentation)

---

## Project Overview

### The Challenge

Deep learning models, especially transformer-based architectures like Wav2Vec2, are computationally expensive. While they achieve excellent accuracy, their inference latency makes real-time deployment challenging.

### Our Solution

This project implements a complete optimization pipeline that:

1. **Profiles** the original PyTorch model to identify bottlenecks
2. **Exports** to ONNX format for framework-agnostic deployment
3. **Optimizes** the ONNX graph (operator fusion, constant folding)
4. **Compiles** to TensorRT for GPU-accelerated inference
5. **Deploys** via NVIDIA Triton Inference Server for production serving
6. **Benchmarks** all stages to quantify improvements

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Backend Support** | PyTorch, ONNX Runtime, TensorRT, Triton |
| **Precision Modes** | FP32, FP16, Mixed Precision |
| **Automated Pipeline** | Single command to run all optimization steps |
| **Comprehensive Logging** | All steps logged for debugging and analysis |
| **Visualization** | Latency plots, operator breakdowns, comparison charts |
| **WER Evaluation** | Word Error Rate computed on Common Voice Hindi |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐    │
│  │ PyTorch  │───▶│   ONNX   │───▶│ TensorRT │───▶│ Triton Server   │   │
│  │  Model   │    │  Export  │    │  Engine  │    │ (Production)     │    │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘    │
│       │               │               │                   │              │
│       ▼               ▼               ▼                   ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐    │
│  │ Profiler │    │ Optimize │    │ FP16/32  │    │ Health Check     │    │
│  │  Stats   │    │  Graph   │    │  Mixed   │    │ Benchmarks       │    │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

The `run_pipeline.py` orchestrates the following steps:

```
Step 1: PyTorch Profiling
    │   └── Identifies expensive operations in the original model
    ▼
Step 2: Export to ONNX
    │   └── Converts PyTorch model to ONNX format (opset 18)
    ▼
Step 3: Optimize ONNX
    │   └── Graph optimizations: fusion, folding, simplification
    ▼
Step 4: Benchmark ONNX Runtime
    │   └── Measures latency with CUDA Execution Provider
    ▼
Step 5: Compute WER
    │   └── Validates accuracy on Common Voice Hindi dataset
    ▼
Step 6: Build TensorRT Engines
    │   └── Creates optimized engines (FP32, FP16, Mixed)
    ▼
Step 7: Deploy to Triton
    │   └── Copies best engine to Triton model repository
    ▼
Step 8: Triton Health Check
    │   └── Validates server is serving correctly
    ▼
Step 9: Compare Results
        └── Generates comparison plots and summary CSV
```

---

## Project Structure

### Root Directory

| File/Folder | Purpose |
|-------------|---------|
| `run_pipeline.py` | **Main entry point** - orchestrates all optimization steps |
| `SETUP.md` | Complete environment setup guide |
| `README.md` | This documentation file |
| `requirements.txt` | Python dependencies (pip) |
| `environment.yml` | Conda environment specification |
| `LICENSE` | Project license |

---

### `code/` - Source Code

All optimization logic is organized into focused modules:

#### `code/profiling/`
| File | Purpose |
|------|---------|
| `pytorch_profiler.py` | Profiles PyTorch model operations, identifies bottlenecks, outputs operator timing CSV |

#### `code/onnx_export/`
| File | Purpose |
|------|---------|
| `export_to_onnx.py` | Exports PyTorch Wav2Vec2 to ONNX format with dynamic axes support |

#### `code/onnx_optim/`
| File | Purpose |
|------|---------|
| `optimize_onnx.py` | Applies ONNX graph optimizations (operator fusion, constant folding, shape inference) |

#### `code/tensorrt/`
| File | Purpose |
|------|---------|
| `build_engine_py.py` | Builds TensorRT engines using Python API (FP32/FP16/Mixed) |
| `benchmark_trt_engine.py` | Benchmarks TensorRT engine latency using CUDA driver API |
| `parse_trt_profile.py` | Parses TensorRT profiling JSON to CSV for analysis |
| `deploy_triton_model.py` | Deploys TensorRT engine to Triton model repository |
| `build_engine_fp16.sh` | Shell script for FP16 engine build via trtexec |
| `build_engine_mixed.sh` | Shell script for mixed precision build via trtexec |
| `benchmark_trt.sh` | Shell script for TRT benchmarking via trtexec |

#### `code/triton/`
| File | Purpose |
|------|---------|
| `triton_check.py` | Validates Triton server health, runs inference, measures latency |
| `analyze_triton.sh` | Shell script for Triton server analysis |
| `model_repository/` | Triton model repository structure |

#### `code/benchmarking/`
| File | Purpose |
|------|---------|
| `benchmark_latency.py` | **Unified benchmarking** - PyTorch, ONNX, TensorRT, Triton |
| `compare_backends.py` | Generates comparison charts and CSV across all backends |
| `compare_ops.py` | Compares operator-level performance across backends |
| `compute_wer.py` | Calculates Word Error Rate on Common Voice Hindi |

#### `code/utils/`
| File | Purpose |
|------|---------|
| `logger.py` | Centralized logging utility - console + file output |

---

### `models/` - Model Artifacts

| Folder | Contents |
|--------|----------|
| `models/pytorch/` | Original PyTorch model state dict |
| `models/onnx/` | ONNX models (raw + optimized) and metadata |
| `models/onnx/backups/` | Timestamped backups of optimized ONNX models |
| `models/tensorrt/` | TensorRT engine files (.plan) for FP32/FP16/Mixed |

**Key Model Files:**

| File | Format | Size | Purpose |
|------|--------|------|---------|
| `wav2vec2_state_dict.pt` | PyTorch | ~380MB | Original model weights |
| `wav2vec2.onnx` | ONNX | ~380MB | Exported ONNX graph |
| `wav2vec2_optimized.onnx` | ONNX | ~380MB | Optimized ONNX (fused ops) |
| `wav2vec2_optimized_trt_fp16.plan` | TensorRT | ~150MB | FP16 optimized engine |
| `wav2vec2_optimized_trt_fp32.plan` | TensorRT | ~380MB | FP32 engine |
| `wav2vec2_optimized_trt_mixed.plan` | TensorRT | ~200MB | Mixed precision engine |

---

### `results/` - Output Artifacts

| File | Description |
|------|-------------|
| `latency_pytorch.txt` | PyTorch inference latency (ms) |
| `latency_onnx.txt` | ONNX Runtime latency (ms) |
| `latency_trt_fp16.txt` | TensorRT FP16 latency (ms) |
| `latency_trt_fp32.txt` | TensorRT FP32 latency (ms) |
| `latency_trt_mixed.txt` | TensorRT Mixed latency (ms) |
| `backend_latency_comparison.csv` | All backends comparison table |
| `backend_latency_comparison.png` | Latency bar chart |
| `onnx_ops.csv` | Top ONNX operators by time |
| `onnx_top_ops.png` | ONNX operator bar chart |
| `pytorch_ops.csv` | PyTorch operator breakdown |
| `WER_hi.txt` | Word Error Rate on Hindi |
| `wer_hi.png` | WER visualization |
| `terminal_logs/` | Detailed logs for each pipeline step |

---

### `triton/` - Triton Inference Server

```
triton/
└── model_repository/
    └── wav2vec2/
        ├── 1/
        │   └── model.plan      # TensorRT engine (deployed)
        └── config.pbtxt        # Triton model configuration
```

**config.pbtxt** specifies:
- Model name: `wav2vec2`
- Backend: `tensorrt`
- Input: `input_values` (FP32, shape: [batch, 16000])
- Output: `logits` (FP32, shape: [batch, seq, vocab])
- Instance group: 1 GPU

---

### `config/`

| File | Purpose |
|------|---------|
| `model.py` | Model configuration constants (model name, paths, hyperparameters) |

---

### `netron/` - Model Visualization

Shell scripts to visualize models using [Netron](https://netron.app/):

| Script | Opens |
|--------|-------|
| `view_pytorch.sh` | PyTorch model graph |
| `view_onnx.sh` | ONNX model graph |
| `view_tensorrt.sh` | TensorRT engine structure |

---

### `Images/` - Documentation Assets

Pre-generated visualizations for documentation:

| Image | Shows |
|-------|-------|
| `Project Pipeline.png` | Overall pipeline architecture |
| `PyTorch Profiling.png` | PyTorch operator profiling results |
| `ONNX Optimisation.png` | Before/after ONNX optimization |
| `ONNX Operation Plots.png` | ONNX operator breakdown |
| `Top-30 ONNX Operations.png` | Most time-consuming ONNX ops |

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/Coder-Abhimanyu/AI4BHARAT-ASR-Model-Optimization.git
cd AI4BHARAT-ASR-Model-Optimization/submission

# Create conda environment
conda env create -f environment.yml
conda activate env-ai4bharat
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

### 2. Run Complete Pipeline

```bash
python run_pipeline.py
```

This executes all optimization steps and generates results in `results/`.

### 3. Run Individual Steps

```bash
# Profile PyTorch model
python code/profiling/pytorch_profiler.py

# Export to ONNX
python code/onnx_export/export_to_onnx.py

# Optimize ONNX
python code/onnx_optim/optimize_onnx.py

# Benchmark all backends
python code/benchmarking/benchmark_latency.py

# Compare results
python code/benchmarking/compare_backends.py
```

### 4. Start Triton Server

```bash
# Via Docker (recommended)
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver --model-repository=/models

# Validate
python code/triton/triton_check.py
```

---

## Results

### Latency Comparison (Example)

| Backend | Latency (ms) | Speedup vs PyTorch |
|---------|--------------|-------------------|
| PyTorch | 45.2 | 1.0x |
| ONNX Runtime | 28.4 | 1.6x |
| TensorRT FP32 | 18.7 | 2.4x |
| TensorRT FP16 | 9.3 | 4.9x |
| TensorRT Mixed | 10.1 | 4.5x |
| Triton (FP16) | 10.5 | 4.3x |

*Note: Results vary based on GPU. Tested on NVIDIA RTX 3060.*

### Word Error Rate

- **Hindi (Common Voice)**: ~15-20% WER
- Accuracy maintained across all optimization stages

---

## Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | Complete environment setup guide |
| [README.md](README.md) | This file - project overview |
| `results/terminal_logs/*.log` | Detailed execution logs |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Base Model** | AI4Bharat IndicWav2Vec (Wav2Vec2) |
| **Framework** | PyTorch 2.9 |
| **Export Format** | ONNX (opset 18) |
| **Optimization** | ONNX Runtime, ONNX Simplifier |
| **Compilation** | NVIDIA TensorRT 10.x |
| **Serving** | NVIDIA Triton Inference Server |
| **GPU** | CUDA 12.x, cuDNN 9.x |

---

## File Execution Order

When `run_pipeline.py` executes, files are called in this order:

```
1. code/profiling/pytorch_profiler.py
2. code/onnx_export/export_to_onnx.py
3. code/onnx_optim/optimize_onnx.py
4. code/benchmarking/benchmark_latency.py
5. code/benchmarking/compute_wer.py
6. code/tensorrt/build_engine_py.py
7. code/tensorrt/benchmark_trt_engine.py (for each engine)
8. code/triton/triton_check.py
9. code/benchmarking/compare_backends.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [AI4Bharat](https://ai4bharat.org/) for the IndicWav2Vec model
- [Hugging Face](https://huggingface.co/) for Transformers library
- [NVIDIA](https://developer.nvidia.com/) for TensorRT and Triton

---

**Built with for Indian Language AI**