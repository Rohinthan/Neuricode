# Neuricode System Architecture Audit

## 1. System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              NEURAICODE RUNTIME ENGINE                               |
|                                                                                   |
|  [ Command Line CLI / TUI ] ---> [ Model Loader (Binary) ]                        |
|                                            │                                      |
|                                            ▼                                      |
|                                 [ Tokenizer Engine ]                              |
|                                            │                                      |
|                                            ▼                                      |
|                             [ Transformer Causal Engine ]                         |
|                                            │                                      |
|                      ┌─────────────────────┴─────────────────────┐                |
|                      ▼                                           ▼                |
|           [ CPU Fallback Engine ]                     [ CUDA GPU Engine ]         |
|        - OpenMP Thread Pool                       - cuBLAS Row-Major GEMMs        |
|        - SIMD Vector Kernels                      - Fused Bias Activations        |
|        - Autograd Memory Tape                     - Dynamic Warp/Block Softmax    |
|                                                   - Workspace Pool Allocator      |
+-----------------------------------------------------------------------------------+
```

## 2. Key Source File Inventory

| Path | Purpose & Responsibilities |
| :--- | :--- |
| [`include/tensor.h`](file: Neuricode/include/tensor.h) | Tensor structure definition, shape metadata, memory pointers, and autograd tape graph nodes. |
| [`src/tensor.c`](file: Neuricode/src/tensor.c) | Tensor allocations (`tensor_create`), gradient accumulation, topological sort, and CPU SIMD operations. |
| [`src/layer.c`](file: Neuricode/src/layer.c) | Dense, CNN, RNN layer abstractions and forward/backward execution dispatch. |
| [`src/transformer.c`](file:Neuricode/src/transformer.c) | Transformer model architecture, RoPE embedding, RMSNorm, SwiGLU FFN, and KV-cache manager. |
| [`cuda/include/cuda_backend.h`](file:Neuricode/cuda/include/cuda_backend.h) | C-callable CUDA API signatures, memory pool declarations, cuBLAS GEMM, and VRAM diagnostics. |
| [`cuda/src/cuda_backend.cu`](file:Neuricode/cuda/src/cuda_backend.cu) | CUDA kernels (`k_softmax_warp`, `k_softmax_block`), cuBLAS calls, and FP16 `cublasGemmEx` launchers. |
| [`cuda/src/cuda_stub.c`](file:Neuricode/cuda/src/cuda_stub.c) | CPU fallback stubs enabling GCC compilation when GPU mode is disabled (`NEURALC_USE_GPU=0`). |
| [`apps/pipeline.c`](file: Neuricode/apps/pipeline.c) | Pipeline execution, autoregressive generation loop, and logit copying. |
| [`src/tokenizer.c`](file: Neuricode/src/tokenizer.c) | BPE/WordPiece token encoding and decoding logic. |
| [`makefile`](file: Neuricode/makefile) | Build System rules for OpenMP, CPU fallback, NVCC compilation, and test targets. |

## 3. Tensor Memory & Layout Model
- **Memory Layout**: Contiguous 1D C `float*` array representing row-major dimensions ($B \times I \times O$).
- **Ownership Model**: Long-lived weights and activations are allocated via `tensor_create()` and freed via `tensor_free()`. Temporary GPU execution scratchpads use the 256-byte aligned bump-pointer pool (`cuda_pool_alloc`).

## 4. CUDA Execution & Stream Model
- **cuBLAS GEMM Mapping**: Applies $C = A \cdot B \iff C^T = B^T \cdot A^T$ to map row-major matrices to column-major cuBLAS without allocation:
  - Forward ($Z = X \cdot W^T + b$): $Z^T = W \cdot X^T$ (`CUBLAS_OP_T`, `CUBLAS_OP_N`)
  - Weight Gradient ($dW = \frac{1}{B} dZ^T \cdot X$): $dW^T = \frac{1}{B} X^T \cdot dZ$ (`CUBLAS_OP_N`, `CUBLAS_OP_T`)
  - Input Gradient ($dX = dZ \cdot W$): $dX^T = W^T \cdot dZ^T$ (`CUBLAS_OP_N`, `CUBLAS_OP_N`)
- **Stream Binding**: Synchronizes cuBLAS handles to active streams via `cublasSetStream(g_cublas, g_active_stream)`.

## 5. Existing Operator Coverage & LLM Gap Analysis

### Existing Operators
- Dense Linear GEMM, Bias Addition, Fused Bias+ReLU, Fused Bias+GELU, Fused Bias+SiLU.
- Elementwise Add, Sub, Mul, Scale, Relu, Sigmoid, Tanh.
- Dynamic Softmax (Warp shuffle for $\text{cols} \le 1024$, Block shared memory for $\text{cols} > 1024$).
- Conv2D, MaxPool2D, Pad2D, Unpad2D.
- FP16 Conversion and FP16 Tensor Core GEMM.
- RMSNorm, RoPE, KV Cache, Tokenizer.

### Missing Components for Production LLM Deployment
- Binary Reader & Tensor Weight Mapper.
- Fused FlashAttention-2 Tiled Attention.
- CUDA Graph Decode Capture (`cudaStreamBeginCapture`).
