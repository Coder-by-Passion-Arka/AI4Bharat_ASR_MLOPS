# 🛠️ Project Setup Guide – AI4Bharat ASR Optimization

This document provides a comprehensive, step-by-step guide to set up the development environment for the **AI4Bharat Speech Recognition Optimization** project.

The project focuses on optimizing a **Wav2Vec2-based ASR model** for deployment on NVIDIA hardware using **ONNX**, **TensorRT**, and **NVIDIA Triton Inference Server**.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Environment Setup](#3-environment-setup)
4. [Install Core Dependencies](#4-install-core-dependencies)
5. [Install PyTorch (CUDA-enabled)](#5-install-pytorch-cuda-enabled)
6. [Install cuDNN](#6-install-cudnn)
7. [Install TensorRT SDK](#7-install-tensorrt-sdk)
8. [Install ONNX Runtime (GPU)](#8-install-onnx-runtime-gpu)
9. [Download Common Voice Dataset](#9-download-common-voice-dataset)
10. [TensorRT Installation for WSL](#10-tensorrt-installation-for-wsl)
11. [Docker Setup for Triton Server](#11-docker-setup-for-triton-server)
12. [Launch Triton Inference Server](#12-launch-triton-inference-server)
13. [Running the Pipeline](#13-running-the-pipeline)
14. [Troubleshooting](#14-troubleshooting)
15. [Verification Checklist](#15-verification-checklist)

---

## 1. Prerequisites

### For smooth setup
- NVIDIA CUDA Version 12.x
- TensorRT Version 8.x
- Conda (Miniconda or Anaconda)
- Python Version 3.10 (Used for this project)
### Optional (but recommended)
- Google Colab (for initial experimentation)
- Docker (for Triton Inference Server)

---

## 2. Create Conda Environment

Create a fresh Conda environment named `env-ai4bharat`:

```bash
# Create a new conda environment
conda create -n env-ai4bharat python=3.10 -y

# Activate the environment
conda activate env-ai4bharat
```

3. Then proceed to install dependencies manually (Steps 4-8).

---

## 4. Install Core Dependencies

If you created the environment manually (Option B), install the core packages:

```bash
conda env create -f environment.yml
```

If you have installed any more dependencies in the conda virtual env, make sure to update the environment.yml file using,
```bash
conda env export > environment.yml
```

### Key Packages Installed

| Package | Purpose |
|---------|---------|
| `transformers` | Pre-trained Wav2Vec2 model |
| `onnx` | ONNX model format support |
| `onnxruntime-gpu` | GPU-accelerated ONNX inference |
| `tensorrt` | NVIDIA TensorRT optimization |
| `tritonclient` | Triton Inference Server client |
| `datasets` | HuggingFace datasets library |
| `evaluate` | WER computation |
| `jiwer` | Word Error Rate metrics |
| `matplotlib` | Visualization |
| `pandas` | Data analysis |

---

## 5. Install PyTorch (CUDA-enabled)

> **⚠️ Important:** Match PyTorch version with your CUDA version!

### For CUDA 12.8 (Recommended)

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

### For CUDA 12.4

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

## 4. Install PyTorch Manually (This version is compatible with CUDA 12.8)

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### Verify PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

> 📘 For other CUDA versions: [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

---

## 6. Install cuDNN

cuDNN provides optimized primitives for deep learning.

### Via Conda (Recommended)

```bash
# For CUDA 12.x
conda install nvidia::cudnn cuda-version=12
```
For any other version of CUDA installed, please download the appropriate version of cuDNN from https://developer.nvidia.com/cudnn. Keeping in mind your native OS, CUDA version and cuDNN version.

## 6. Install TensorRT SDK Manually
For Linux / WSL Conda enviroment:
```bash
python.exe -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12
```
For any other Operating System, please download the appropriate version of TensorRT from https://developer.nvidia.com/tensorrt/download. Keeping in mind your native Python version, CUDA version and TensorRT version.

## 7. Download and extract Common Voice Dataset
For some reason the Common-Voice Dataset in HuggingFace is Deprecated. 
And the Mozilla.Org has pulled back the dataset from all Thrid Party Dataset providers. 

To download the dataset, you have to go to: https://datacollective.mozillafoundation.org/datasets

Register yourself, make an API key and download the dataset you want to download using the API key.

```bash
# Set your API key
export MDC_API_KEY="YOUR_API_KEY_HERE"

# Request download URL
RESPONSE=$(curl -s -X POST \
  "https://datacollective.mozillafoundation.org/api/datasets/cmj8u3p7m00b1nxxbm34a993r/download" \
  -H "Authorization: Bearer $MDC_API_KEY" \
  -H "Content-Type: application/json")

# Check response
echo "$RESPONSE" | jq .

# Extract download URL
DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')

# Validate URL
if [[ "$DOWNLOAD_URL" == "null" || -z "$DOWNLOAD_URL" ]]; then
  echo "Failed to get download URL"
  exit 1
fi

# Download and extract
curl -L "$DOWNLOAD_URL" -o "CommonVoice_Hindi.tar.gz"

# 7. Extract downloaded dataset
tar -xvf CommonVoice_Hindi.tar.gz -C data/common_voice_/hi/
```

## 8. TensorRT Installation for WSL
This method of installation is recommended for WSL2 only.
Make sure you download a compatible version. Otherwise this can corrupt the entire pipeline.

```bash
sudo apt update

sudo apt -o Acquire::Retries=5 \
         -o Acquire::http::Timeout=60 \
         -o Acquire::https::Timeout=60 \
         install --fix-missing \
         libnvinfer10 \
         libnvinfer-dev \
         libnvinfer-plugin10 \
         libnvinfer-plugin-dev \
         tensorrt
```

### Verify Installation

```bash
# Check trtexec location
which trtexec

# Check version
trtexec --version

# Verify GPU access
nvidia-smi
```
How ever, some versions of 'trtexec' will show CUDA compatibility issues if CUDA drivers and toolkit is under 13.x.x version.

Thus I have kept TensorRT API calling in case 'trtexec' fails.

## 9. Launch Triton Inference Server
```bash
tritonserver \
  --model-repository=triton/model_repository \
  --strict-model-config=false
```

## 10. The ONNX Optimisation may lead to WSL Faulty Exit

This is happening because we are using some functions that optimised model at onnx-runtime
To recover from this, you have to Edit (Or Create if not already present) the .wslconfig file 
Open PowerShell to perform these steps

1. Migrate to Home Folder
```powershell
cd~
```

2. Check if the File already exists or not 
``` powershell 
Test-Path "$HOME\.wslconfig"
```
3. Create and Open the .wslconfig File 

```powershell
# Navigate to home directory
cd ~

# Create the config file
New-Item -Path "$HOME\.wslconfig" -ItemType File -Force

# Open in Notepad
notepad "$HOME\.wslconfig"
```

4. Required File Content (Example)
Paste the following block into Notepad. Ensure there are no typos in the headers (e.g., [wsl2]).
```ini
[wsl2]
memory=16GB
processors=8
swap=16GB
```

Action: Save (Ctrl+S) and close Notepad.

5. Restart WSL to Apply Changes
WSL 2 only reads the .wslconfig file on a cold boot. You must shut down the WSL engine.
powershell
```powershell
wsl --shutdown
```

4. Verify File Integrity (PowerShell)
Confirm the file exists in the correct location and verify that the content was saved successfully.
```powershell
# Verify the absolute path
Resolve-Path "$HOME\.wslconfig"

# Verify the content is correctly written
Get-Content "$HOME\.wslconfig"
```

5. Verify Active Configuration (Linux Terminal)

To confirm that the WSL kernel has actually adopted the new limits, run these commands via the wsl prefix.
powershell

```bash
# Check total available RAM (should be ~15-16Gi)
wsl free -h

# Check available CPU cores (should be 8)
wsl nproc

# Verify you are running WSL version 2
wsl -l -v
```

---

### ONNX Runtime GPU Not Available

```bash
# Check available providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# If 'CUDAExecutionProvider' is missing:
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

---

### Triton Server Connection Refused

```bash
# Check if Triton is running
docker ps

# Check Triton logs
docker logs <container_id>

# Verify model repository structure
ls -la triton/model_repository/wav2vec2/
# Should contain: 1/model.plan, config.pbtxt
```

---

### Import Errors ("No module named 'code'")

The `code` directory conflicts with Python's built-in `code` module.

**Solution:** The pipeline uses `importlib` for dynamic imports, but ensure you run scripts from the project root:

```bash
# Always run from submission/ directory
cd /path/to/submission
python run_pipeline.py
```

---

## 15. Verification Checklist

Use this checklist to verify your setup is complete:

### Environment
- [ ] Conda environment `env-ai4bharat` is active
- [ ] Python version is 3.10.x

### GPU Stack
- [ ] `nvidia-smi` shows GPU
- [ ] PyTorch detects CUDA: `torch.cuda.is_available() == True`
- [ ] ONNX Runtime has GPU: `'CUDAExecutionProvider' in ort.get_available_providers()`

### TensorRT
- [ ] `import tensorrt` works without errors
- [ ] `trtexec --version` runs (or Python API works as fallback)

### Docker (for Triton)
- [ ] `docker --version` shows version
- [ ] `docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi` shows GPU

### Project Files
- [ ] `models/onnx/wav2vec2.onnx` exists
- [ ] `models/tensorrt/*.plan` engines exist
- [ ] `triton/model_repository/wav2vec2/1/model.plan` exists

### Quick Verification Script

```bash
# Run from submission/ directory
python -c "
import torch
import onnxruntime as ort
import tensorrt as trt

print('=== Environment Verification ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'ONNX Runtime: {ort.__version__}')
print(f'ONNX Providers: {ort.get_available_providers()}')
print(f'TensorRT: {trt.__version__}')
print('=== All checks passed! ===')
"
```

---

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [AI4Bharat IndicWav2Vec](https://ai4bharat.org/)

---

## 🤝 Contributing

If you encounter issues not covered in this guide, please:

1. Check the [Troubleshooting](#14-troubleshooting) section
2. Review the terminal logs in `results/terminal_logs/`
3. Open an issue on the GitHub repository

---

**Happy Optimizing! 🚀**
