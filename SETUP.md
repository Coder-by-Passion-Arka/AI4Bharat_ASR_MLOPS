# Project Setup Guide – AI4Bharat ASR Optimization

This document explains how to set up the development environment for the
AI4Bharat Speech Recognition Optimization project.

The project focuses on optimizing a Wav2Vec2-based ASR model for deployment
on NVIDIA hardware using ONNX and TensorRT.

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
conda create -n env-ai4bharat python=3.10 -y
conda activate env-ai4bharat
```

## 3. Install Dependencies automatically

```bash
pip install -r requirements.txt
```

## 4. Install PyTorch Manually (This version is compatible with CUDA 12.8)

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```
For any other version of CUDA installed, please download the appropriate version of PyTorch from https://pytorch.org/get-started/previous-versions/

## 5. Install cuDNN

```bash
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
# Get download URL to download file
# 1. Set API key
export MDC_API_KEY="API_KEY"

# 2. Request a download URL
RESPONSE=$(curl -s -X POST \
  "https://datacollective.mozillafoundation.org/api/datasets/cmj8u3p7m00b1nxxbm34a993r/download" \
  -H "Authorization: Bearer $MDC_API_KEY" \
  -H "Content-Type: application/json")

# 3. Inspect response (IMPORTANT for debugging)
echo "$RESPONSE" | jq .

# 4. Extract download URL
DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl')

# 5. Validate URL
if [[ "$DOWNLOAD_URL" == "null" || -z "$DOWNLOAD_URL" ]]; then
  echo "Failed to get download URL"
  exit 1
fi

# 6. Download dataset
curl -L "$DOWNLOAD_URL" -o "CommonVoice_Hindi.tar.gz"

# 7. Extract downloaded dataset
tar -xvf CommonVoice_Hindi.tar.gz -C data/common_voice_/hi/
```

## 8. TensorRT Installation for WSL
This method of installation is recommended for WSL2 only.

```bash
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

Verification after installation:
```bash
trtexec --version
nvidia-smi
```
## 9. Launch Triton Inference Server
```bash
tritonserver \
  --model-repository=triton/model_repository \
  --strict-model-config=false
```