#!/usr/bin/env python3
"""
Benchmark a TensorRT engine using the Python API.

Latency semantics:
- All timings are measured in SECONDS internally
- All reported & written latencies are in MILLISECONDS (ms)

Output:
- A single float written to OUT_TXT (milliseconds)
"""
#!/usr/bin/env python3
import sys
import time
import numpy as np
from pathlib import Path

import tensorrt as trt
from cuda import cuda

# Custom logger import - assuming this exists in your repo
try:
    from code.utils.logger import get_logger
    logger = get_logger("benchmark_trt_engine", level=10)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("benchmark_trt_engine")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
N_WARMUP = 5
N_RUNS = 20

# -----------------------------------------------------------------------------
# def check(status):
#     """
#     Unpacks the CUresult from cuda-python calls and raises error if not success.
#     """
#     err = status[0] if isinstance(status, tuple) else status
#     if err != cuda.CUresult.CUDA_SUCCESS:
#         raise RuntimeError(f"CUDA Driver Error: {err}")
#     # Return the value part of the tuple if it exists, else just the error code
#     return status[1] if isinstance(status, tuple) else status

def check(status):
    # Unpack the error code (always at index 0)
    err = status[0] if isinstance(status, tuple) else status
    
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA Driver Error: {err}")
    
    # If there's a second element (the actual data/pointer), return it
    if isinstance(status, tuple) and len(status) > 1:
        return status[1]
    
    # Otherwise return the error code/success status
    return err

# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python -m code.tensorrt.benchmark_trt_engine <engine.plan> <out_latency.txt>")
        sys.exit(1)

    # --- Initialize CUDA Driver (CRITICAL for Driver API) ---
    check(cuda.cuInit(0))
    _, dev = cuda.cuDeviceGet(0)
    _, primary_ctx = cuda.cuDevicePrimaryCtxRetain(dev)
    check(cuda.cuCtxPushCurrent(primary_ctx))

    engine_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    logger.info(f"Benchmarking TensorRT engine → {engine_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # --- Load engine
    with engine_path.open("rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    logger.info("Engine deserialized")
    context = engine.create_execution_context()

    # --- Identify tensors
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) 
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) 
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    inp, out = input_names[0], output_names[0]

    # --- Define input shape
    input_shape = (1, 16000)
    context.set_input_shape(inp, input_shape)

    # --- Allocate buffers using Driver API (cuMemAlloc)
    input_nbytes = np.prod(input_shape) * np.dtype(np.float32).itemsize
    output_shape = tuple(context.get_tensor_shape(out))
    output_nbytes = np.prod(output_shape) * np.dtype(np.float32).itemsize

    d_input = check(cuda.cuMemAlloc(input_nbytes))
    d_output = check(cuda.cuMemAlloc(output_nbytes))

    context.set_tensor_address(inp, int(d_input))
    context.set_tensor_address(out, int(d_output))

    # Create stream (cuStreamCreate)
    stream = check(cuda.cuStreamCreate(0))

    # --- Prepare input
    host_input = np.random.randn(*input_shape).astype(np.float32)
    # Use cuMemcpyHtoDAsync for Driver API
    check(cuda.cuMemcpyHtoDAsync(
        d_input,
        host_input.ctypes.data,
        input_nbytes,
        stream
    ))

    # --- Warmup
    logger.info(f"Warmup: {N_WARMUP} runs")
    for _ in range(N_WARMUP):
        context.execute_async_v3(stream)
    check(cuda.cuStreamSynchronize(stream))

    # --- Timed runs
    logger.info(f"Timed runs: {N_RUNS}")
    times = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        context.execute_async_v3(stream)
        check(cuda.cuStreamSynchronize(stream))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = float(np.mean(times) * 1000.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{avg_ms:.6f}")

    logger.info(f"TensorRT average latency: {avg_ms:.3f} ms")

    # --- Cleanup
    check(cuda.cuMemFree(d_input))
    check(cuda.cuMemFree(d_output))
    check(cuda.cuStreamDestroy(stream))
    cuda.cuDevicePrimaryCtxRelease(dev)

if __name__ == "__main__":
    main()

# import sys
# import time
# import numpy as np
# from pathlib import Path

# import tensorrt as trt
# from cuda import cuda

# from code.utils.logger import get_logger

# logger = get_logger("benchmark_trt_engine", level=10)

# # -----------------------------------------------------------------------------
# # Config
# # -----------------------------------------------------------------------------
# N_WARMUP = 5
# N_RUNS = 20

# # -----------------------------------------------------------------------------
# # def check(status):
# #     if status[0] != cuda.cudaError_t.cudaSuccess:
# #         raise RuntimeError(f"CUDA error: {status}")

# def check(status):
#     from cuda import cuda
    
#     # Unpack the error code if status is a tuple (e.g., from cuMemAlloc)
#     err = status[0] if isinstance(status, tuple) else status
    
#     # Driver API success check
#     if err != cuda.CUresult.CUDA_SUCCESS:
#         raise RuntimeError(f"CUDA error: {err}")
    
#     return status

# # -----------------------------------------------------------------------------
# def main():
#     if len(sys.argv) != 3:
#         print("Usage:")
#         print("  python -m code.tensorrt.benchmark_trt_engine <engine.plan> <out_latency.txt>")
#         sys.exit(1)

#     engine_path = Path(sys.argv[1])
#     out_path = Path(sys.argv[2])

#     if not engine_path.exists():
#         raise FileNotFoundError(engine_path)

#     logger.info(f"Benchmarking TensorRT engine → {engine_path}")

#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#     # --- Load engine
#     with engine_path.open("rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())

#     logger.info("Engine deserialized")

#     context = engine.create_execution_context()

#     # --- Identify tensors
#     input_names = []
#     output_names = []

#     for i in range(engine.num_io_tensors):
#         name = engine.get_tensor_name(i)
#         if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
#             input_names.append(name)
#         else:
#             output_names.append(name)

#     logger.info(f"TensorRT inputs : {input_names}")
#     logger.info(f"TensorRT outputs: {output_names}")

#     assert len(input_names) == 1, "Expected single input"
#     assert len(output_names) == 1, "Expected single output"

#     inp = input_names[0]
#     out = output_names[0]

#     # --- Define input shape (batch=1, 16000 samples)
#     input_shape = (1, 16000)
#     context.set_input_shape(inp, input_shape)

#     assert context.all_binding_shapes_specified

#     # --- Allocate buffers
#     input_nbytes = np.prod(input_shape) * np.dtype(np.float32).itemsize
#     output_shape = tuple(context.get_tensor_shape(out))
#     output_nbytes = np.prod(output_shape) * np.dtype(np.float32).itemsize

#     check(cuda.cuMemAlloc(input_nbytes))
#     d_input = check(cuda.cuMemAlloc(input_nbytes))[1]
#     d_output = check(cuda.cuMemAlloc(output_nbytes))[1]

#     context.set_tensor_address(inp, d_input)
#     context.set_tensor_address(out, d_output)

#     # check(cuda.cudaStreamCreate())
#     # stream = check(cuda.cudaStreamCreate())[1]

#     err, stream = cuda.cuStreamCreate(0)
#     check(err)

#     # --- Prepare input
#     host_input = np.random.randn(*input_shape).astype(np.float32)
#     check(cuda.cudaMemcpyAsync(
#         d_input,
#         host_input.ctypes.data,
#         input_nbytes,
#         cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
#         stream
#     ))

#     # --- Warmup
#     logger.info(f"Warmup: {N_WARMUP} runs")
#     for _ in range(N_WARMUP):
#         context.execute_async_v3(stream.handle)
#     check(cuda.cudaStreamSynchronize(stream))

#     # --- Timed runs
#     logger.info(f"Timed runs: {N_RUNS}")
#     times = []

#     for _ in range(N_RUNS):
#         t0 = time.perf_counter()
#         context.execute_async_v3(stream.handle)
#         check(cuda.cudaStreamSynchronize(stream))
#         t1 = time.perf_counter()
#         times.append(t1 - t0)

#     avg_ms = float(np.mean(times) * 1000.0)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(f"{avg_ms:.6f}")

#     logger.info(f"TensorRT average latency: {avg_ms:.3f} ms")
#     logger.info(f"Wrote latency → {out_path}")

#     # --- Cleanup (important!)
#     cuda.cudaFree(d_input)
#     cuda.cudaFree(d_output)
#     cuda.cudaStreamDestroy(stream)

# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()