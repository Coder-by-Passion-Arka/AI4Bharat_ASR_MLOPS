## Project Directory Structure
.
submission
├── code
│   ├── benchmarking
│   │   ├── benchmark_latency.py
│   │   ├── compare_backends.py
│   │   ├── compare_ops.py
│   │   └── compute_wer.py
│   ├── onnx_export
│   │   └── export_to_onnx.py
│   ├── onnx_optim
│   │   └── optimize_onnx.py
│   ├── profiling
│   │   └── pytorch_profiler.py
│   ├── tensorrt
│   │   ├── benchmark_trt.sh
│   │   ├── build_engine_fp16.sh
│   │   ├── build_engine_mixed.sh
│   │   └── parse_trt_profile.py
│   └── triton
│       ├── analyze_triton.sh
│       └── model_repository
│           ├── wav2vec2
│           │   └── 1
│           │       └── model.plan   
│           └── config.pbtxt
├── colab_notebook.ipynb
├── config
│   └── model.py
├── models
│   ├── onnx
│   │   ├── wav2vec2.onnx
│   │   ├── wav2vec2.onnx.data
│   │   └── wav2vec2_optimized.onnx
│   └── pytorch
│       └── wav2vec2_state_dict.pt
├── netron
│   ├── view_onnx.sh
│   ├── view_pytorch.sh
│   └── view_tensorrt.sh
├── requirements.txt
├── results
│   ├── WER_hi.txt
│   ├── onnx_ops.csv
│   ├── onnx_profile.json
│   ├── pytorch_ops.csv
│   └── pytorch_profiler.txt
└── run_pipeline.py
