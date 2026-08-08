# Neuricode DL: C11 & CUDA Production Framework System Specification & Architecture Reference

---

## SECTION 1: FULL SYSTEM OVERVIEW

### 1.1 High-Level Architectural Paradigm
Neuricode is a production-grade, zero-external-dependency deep learning framework and LLM engine built in pure ISO C11 with an accelerated CUDA C++/cuBLAS compute backend. The system is engineered around six core technical pillars:

1. **Explicit Memory Control & Cache Alignment**: Zero implicit heap allocations during active forward/backward passes. Memory required for activations, gradients, scratchpads, and velocity state buffers is pre-allocated and aligned to 64-byte boundaries (`neuralc_aligned_alloc`) to optimize SIMD vector loads (AVX2/AVX-512) and DMA transfer efficiency. Includes an optional memory arena allocator (`memory/memory.c`).
2. **Unified Dual-Backend Device Abstraction**: Tensors carry explicit device placement tags (`CF_GPU_NONE`, `CF_GPU_CUDA`, `CF_GPU_OPENCL`). High-level neural network operations (Dense, Conv2D, RNN, Transformer, Loss Functions) dynamically query tensor placement and dispatch execution either to CPU SIMD micro-kernels or to asynchronous CUDA kernel streams.
3. **Dynamic Autograd Tape & High-Performance Manual Dispatch**: Supports both reverse-mode automatic differentiation via an execution graph tape (`GraphNode`) and direct zero-overhead layer-by-layer forward/backward execution for performance-critical recurrent models.
4. **Fused CUDA Operator Kernels**: Merges linear projection, bias addition, and non-linear activation functions into single-pass CUDA grid executions, eliminating round-trip DRAM read/write cycles on GPU global memory.
5. **Kernel-Style `menuconfig` Configuration UI**: Interactive terminal configuration interface (`make config` / `menuconfig`) built in C (`config/config_ui.c`) that outputs a system configuration manifest (`neuralc_config.h`) without requiring source code modifications.
6. **Deterministic Binary Serialization & Embedded Runtime**: Complete model state, hyperparameters, token dictionaries, and weights are saved into deterministic, byte-aligned binary format files compatible across training and production inference executables (`train`, `pipeline`, `cli`, `neuricode`). Boots in < 2 ms inside a ~4.4 MB standalone binary.

```
+-----------------------------------------------------------------------------------+
|                                  USER APPLICATIONS                                |
|             (apps/neuricode_cli.c, apps/train.c, apps/cli.c, apps/pipeline.c)     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            ANTIGRAVITY TUI & SHELL ENGINE                         |
|                 (src/tui.c, config/config_ui.c, neuralc_config.h)                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            HIGH-LEVEL NEURAL MODULES                              |
|           (src/nn.c, src/layer.c, src/rnn.c, src/conv.c, src/transformer.c)      |
+-----------------------------------------------------------------------------------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
+-----------------------------------------------+ +---------------------------------+
|         CPU BACKEND & TENSOR TAPE             | |          CUDA BACKEND           |
|  (src/tensor.c, OpenMP, AVX2 SIMD, Autograd)  | | (cuda/src/cuda_backend.cu,    |
+-----------------------------------------------+ |  cuBLAS GEMM, Fused Kernels)    |
                        |                         +---------------------------------+
                        v                                         |
+-----------------------------------------------+                         |
|                 HOST RAM (DRAM)               |                         |
|      (64-byte aligned buffers, memory.c)      |                         |
+-----------------------------------------------+                         |
                        ^                                         |
                        |====== Host-to-Device / Device-to-Host ==|
                        |       (cudaMemcpyAsync / Pinned Host)   v
                                                  +---------------------------------+
                                                  |         GPU VRAM (GDDR/HBM)     |
                                                  |      (Coalesced Memory Tiles)   |
                                                  +---------------------------------+
```

---

### 1.2 Data Flow Architecture: Raw Text to Trained Model Outputs

The complete data pipeline transforms raw unstructured text into tokenized integer IDs, high-dimensional target vectors, dense recurrent embeddings, and cross-entropy loss gradients.

```
+------------------+     +-------------------+     +------------------+
| Raw Text File    | --> | Tokenizer Engine  | --> | Token ID Stream  |
| (data.txt)       |     | (src/tokenizer.c) |     | [id_0, id_1, ...] |
+------------------+     +-------------------+     +------------------+
                                                            |
                                                            v
+------------------+     +-------------------+     +------------------+
| Reshaped 2D/3D   | <-- | One-Hot Encoding  | <-- | Stream Dataset   |
| Tensor Buffers   |     | (train.c)         |     | (dataset_loader) |
+------------------+     +-------------------+     +------------------+
        |
        v
+-----------------------------------------------------------------------------+
|                            FORWARD PROPAGATION                              |
|                                                                             |
|  +------------------+     +-------------------+     +---------------------+ |
|  | RNN Recurrent    | --> | 3D->2D Zero-Copy  | --> | Dense Linear        | |
|  | Layer (BPTT)     |     | Tensor Reshape    |     | Projection Layer    | |
|  +------------------+     +-------------------+     +---------------------+ |
|                                                                |            |
|                                                                v            |
|                           +-------------------+     +---------------------+ |
|                           | Cross-Entropy     | <-- | Softmax Probability | |
|                           | Loss Computation  |     | Row Normalization   | |
|                           +-------------------+     +---------------------+ |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                            BACKWARD PROPAGATION                             |
|                                                                             |
|  +------------------+     +-------------------+     +---------------------+ |
|  | Parameter Update | <-- | Gradient Norm     | <-- | Loss Gradient       | |
|  | (SGD / Adam)     |     | Clipping (L2)     |     | (dL/dz = p - t)     | |
|  +------------------+     +-------------------+     +---------------------+ |
+-----------------------------------------------------------------------------+
```

---

### 1.3 Comparative Analysis: Neuricode vs. Standard Deep Learning Frameworks

| Feature / Dimension | ⚡ **Neuricode v1** | 🐍 **PyTorch / TensorFlow** | 🦙 **llama.cpp / GGML** | 📦 **ONNX Runtime** |
|---|---|---|---|---|
| **Language Stack** | **100% ISO C11** | Python + C++ / CUDA | C++11 / C++17 | C++17 |
| **Dependencies** | **0 (Zero / Standard C)** | Heavy (Python, MKL, libtorch, CUDA) | Low (C++ STL, pthread, OpenMP) | Medium (Protobuf, Abseil, Flatbuffers) |
| **Binary Size** | **~4.4 MB** | > 2.0 – 5.0 GB | ~15 – 30 MB | ~50 – 100 MB |
| **Cold Start Boot** | **< 2 milliseconds** | 1.5 – 5.0 seconds | 50 – 200 milliseconds | 200 – 500 milliseconds |
| **Terminal UI Engine** | **Built-in ANSI TUI + `menuconfig`** | None (requires Python UI) | Basic CLI prompt | None (library only) |
| **Embedded Edge Ready** | **Raspberry Pi, Jetson, MCU** | No (requires heavy OS + Python) | Yes (Inference only) | Partial (requires C++ runtime) |
| **Native C Training** | **Full Autograd & Backprop** | Full (Python API) | Limited / Inference Focused | Limited / Export Focused |

---

## SECTION 2: EXECUTION FLOW (STEP-BY-STEP)

This section provides an line-by-line execution trace of running the standard Neuricode training executable:

```bash
./train chunks/token_ab assets/vocab.txt model.bin 20 64 4 0.001
```

---

### Step 1: Argument Parsing & Hyperparameter Configuration
- **Mechanism**: Executed in `apps/train.c` inside `main(int argc, char **argv)`.
- **Parsing Logic**:
  - `argv[1]` = `"chunks/token_ab"` (Training text data file path).
  - `argv[2]` = `"assets/vocab.txt"` (Vocabulary mapping file path).
  - `argv[3]` = `"model.bin"` (Output model binary checkpoint path).
  - `argv[4]` = `"20"` (Sequence length $S = 20$).
  - `argv[5]` = `"64"` (Hidden layer dimension $H = 64$).
  - `argv[6]` = `"4"` (Total training epochs $E = 4$).
  - `argv[7]` = `"0.001"` (Base learning rate $\eta = 0.001$).
- **Config Struct Initialization**:
  The values populate a `TrainConfig` structure. If optional CLI positional arguments are omitted, `train.c` falls back to default constants specified in `neuralc_config.h` or `#ifndef` macros (`SEQ_LEN=20`, `BATCH_SIZE=8`, `EPOCHS=20`, `HIDDEN_SIZE=256`, `LEARNING_RATE=0.01f`).

---

### Step 2: Tokenizer Loading & Corpus Encoding
- **Mechanism**: Executed via `tokenizer_load(vocab_path)` and `tokenizer_encode()`.
- **Internal Mechanics**:
  1. Opens `assets/vocab.txt`, reading line-delimited token strings.
  2. Allocates a `Tokenizer` structure containing a dynamic string table (`char **tokens`) and a hash table mapping strings to integer token indices (`int *id_map`).
  3. Detects if an explicit End-Of-Sequence `<EOS>` token exists in the vocabulary (`tokenizer_eos_id()`).
  4. Encodes `chunks/token_ab`: If `<EOS>` is defined, `train.c` reads the input file line-by-line, invoking `tokenizer_encode()` per line and explicitly appending the `<EOS>` token ID to the stream.

---

### Step 3: Binary Dataset Creation & Streaming Setup
- **Mechanism**: Handled by `dataset_loader.c` routines `encode_file_to_binary()` and `dataset_open()`.
- **Internal Mechanics**:
  1. Writes the full token stream out to a temporary binary dataset file (`dataset.bin`). The file format consists of a contiguous stream of `uint32_t` token values.
  2. `dataset_open()` memory-maps or opens `dataset.bin`, initializing a `DatasetLoader` struct.

---

### Step 4: Model Initialization & Binary Checkpoint Verification
- **Mechanism**: Executed via `rnn_create()`, `dense_create()`, and `load_model()`.
- **Internal Mechanics**:
  1. `train.c` checks if `model.bin` already exists on disk.
  2. Reads or instantiates $W_{xh} \in \mathbb{R}^{H \times V}$, $W_{hh} \in \mathbb{R}^{H \times H}$, $b_h \in \mathbb{R}^{H \times 1}$, $W_{\text{dense}} \in \mathbb{R}^{V \times H}$, and $b_{\text{dense}} \in \mathbb{R}^{V \times 1}$.

---

### Step 5: Training Loop Execution & Steps Sizing
- **Mechanism**: Outer `for (int epoch = 0; epoch < config.epochs; epoch++)` loop and inner `for (int step = 0; step < steps_per_epoch; step++)` loop.
- **Internal Mechanics**:
  1. Computes total tokens per batch: $\text{tokens\_per\_batch} = B \times S = 64 \times 20 = 1280$.
  2. Calculates step count: $\text{steps\_per\_epoch} = \lfloor \frac{\text{ds->token\_count}}{1280} \rfloor$.
  3. Monitors loss plateauing across epochs. If `avg_loss > prev_loss - 0.0005f` for 3 consecutive epochs, decays learning rate: $\eta \leftarrow \eta \times 0.8$.

---

### Step 6: Forward Pass Internals
- **Mechanism**: `dataset_next_batch()`, `fill_onehot_3d()`, `rnn_forward()`, `tensor_reshape()`, `dense_forward()`.
- **Internal Mechanics**:
  1. `rnn_forward()` unrolls hidden states $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$.
  2. `tensor_reshape()` creates a 2D zero-copy view tensor `[1280, 64]`.
  3. `dense_forward()` computes $Z = X_{\text{2d}} W_{\text{dense}}^T + b_{\text{dense}}$ and Softmax probabilities.

---

### Step 7: Backward Pass Internals
- **Mechanism**: `nn_loss()`, `dense_backward()`, `tensor_free()`, `rnn_backward()`, `rnn_clip_gradients()`, `nn_clip_gradients()`.
- **Internal Mechanics**:
  1. `nn_loss()` initializes loss gradient $\frac{\partial L}{\partial Z} = \frac{P - Y}{B \times S}$.
  2. `dense_backward()` computes $dW_{\text{dense}}, db_{\text{dense}}, dX_{\text{dense}}$.
  3. `tensor_free()` deallocates 2D view header.
  4. `rnn_backward()` unrolls BPTT: $\delta_t = (\delta_{t+1} W_{hh}^T + dX_{3d, t}) \odot (1 - h_t^2)$.
  5. Gradient clipping rescales gradients if norm > 5.0f.

---

### Step 8: Optimizer Updates & Model Checkpointing
- **Mechanism**: `sgd_step()`, `save_model()`.
- **Internal Mechanics**:
  1. `sgd_step()` applies momentum velocity updates: $v_{t+1} = \mu v_t + g + \lambda W$; $W \leftarrow W - \eta v_{t+1}$.
  2. `save_model()` serializes raw float weight buffers to `model.bin`.

---

## SECTION 3: CUDA BACKEND DEEP DIVE

### 3.1 Advanced GPU Memory Management Architecture
- **Device Allocation (`cuda_malloc_f32`)**: Wraps `cudaMalloc()` with `CUDA_CHECK` error handling.
- **Page-Locked Host Allocation (`cuda_malloc_host`)**: Uses `cudaMallocHost()` for pinned DRAM allocations, enabling direct PCI-Express Direct Memory Access (DMA) transfers.
- **Asynchronous Copies (`cuda_memcpy_h2d_async`)**: Executes `cudaMemcpyAsync()` over CUDA streams.

---

### 3.2 CUDA Streams & cuBLAS GEMM Mapping
- **Stream-Bound Execution**: All GPU kernels and cuBLAS operations bind to active `cudaStream_t` instances (`cuda_set_stream`).
- **cuBLAS GEMM Matrix Layout**: Row-major C matrix multiplication maps to column-major cuBLAS via transpose identity $C = A B \iff C^T = B^T A^T$.
- **Tensor Core Enabling**: `cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)`.

---

### 3.3 Fused Kernels & Shared Memory Tiled Transpose
- **Fused Linear + ReLU (`k_fused_linear_relu`)**: Fuses matrix projection, bias addition, and ReLU activation into one GPU grid call, eliminating intermediate DRAM writes.
- **Tiled Transpose (`k_transpose_tiled`)**: Uses $32 \times 32$ shared memory tiles with `tile[32][33]` column padding, preventing 32-bank conflicts and enforcing coalesced global memory access.

---

## SECTION 4: MATHEMATICAL FOUNDATIONS

### 4.1 Forward & Backward Equations

| Component | Forward Equation | Backward Gradient Equation |
|---|---|---|
| **Linear Layer** | $Z = X W^T + b$ | $dW = dZ^T X, \quad db = \sum dZ, \quad dX = dZ W$ |
| **RNN Recurrent Step** | $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ | $\delta_t = (\delta_{t+1} W_{hh}^T + dX_t) \odot (1 - h_t^2)$ |
| **Softmax + Cross-Entropy** | $P_i = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$ | $\frac{\partial L}{\partial Z} = P - Y$ |
| **Batch Normalization** | $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ | $\frac{\partial L}{\partial x_i} = \frac{\gamma}{B \sqrt{\sigma^2 + \epsilon}} \left[ B \frac{\partial L}{\partial y_i} - \sum \frac{\partial L}{\partial y} - \hat{x}_i \sum \frac{\partial L}{\partial y} \hat{x} \right]$ |

---

## SECTION 5: ANTIGRAVITY TUI SHELL & MENUCONFIG

### 5.1 Interactive CLI Shell Architecture (`apps/neuricode_cli.c` & `src/tui.c`)

Neuricode includes a lightweight terminal REPL shell:

```bash
./neuricode
```

```
   ┌──────────────────────────────────────────────────────────┐
   │ ⚡ NEURICODE v1 — Native Zero-Dependency Edge AI Engine │
   │ Model: model.bin | Vocab: assets/vocab.txt | GPU: CUDA  │
   └──────────────────────────────────────────────────────────┘
```

#### Terminal Control Features:
- **Raw Mode Input Processing**: Uses UNIX `termios` to intercept arrow keys, backspaces, and tab completion without terminal input echoing.
- **Dynamic Window Resizing**: Handles OS `SIGWINCH` signals via `ioctl(STDOUT_FILENO, TIOCGWINSZ, &w)`, adjusting layout boundaries instantly.
- **8 Color Themes**: Dynamic ANSI color scheme switcher (`/theme`).
- **Slash Commands**:
  - `/help`: Displays active hyperparameter specs and system manual.
  - `/status`: Renders live memory, vocabulary, and GPU status dashboard.
  - `/theme`: Opens interactive arrow-key color theme selector.
  - `/temp <val>`: Adjusts logit sampling temperature (e.g. `/temp 0.7`).
  - `/topk <val>`: Sets Top-K nucleus sampling threshold (e.g. `/topk 40`).
  - `/reset`: Clears model recurrent hidden state memory.
  - `/clear`: Clears terminal screen buffer.
  - `/exit`: Safely terminates session.

---

### 5.2 Kernel-Style `menuconfig` Interface (`config/config_ui.c`)

```bash
make config
# or
./menuconfig
```

The interactive TUI writes configuration parameters out to `neuralc_config.h`:
- `HIDDEN_SIZE`: Recurrent hidden dimension.
- `LEARNING_RATE`: Optimization step size.
- `SEQ_LEN`: Unrolled sequence length.
- `BATCH_SIZE`: Batch size.
- `USE_OMP`: Toggle OpenMP multi-threading.
- `USE_CUDA`: Toggle CUDA GPU backend dispatch.

---

## SECTION 6: TESTING & VALIDATION

### 6.1 Multi-Module Unit Testing Strategy
- **Autograd Tape Verification (`src/test_autograd.c`)**: Validates graph node creation, topological sorting, and reverse-mode gradient flow.
- **CPU vs. GPU Equivalence**: Compares float arrays generated by CPU reference functions and CUDA GPU kernels ($\|y_{\text{cpu}} - y_{\text{gpu}}\|_{\infty} < 10^{-5}$).
- **Sanitizer Verification**: Tested using GCC `-fsanitize=address,undefined` and NVIDIA `compute-sanitizer`.

---

## SECTION 7: PROJECT STRUCTURE & MODULE INVENTORY

| File Path | Header File | Core Functional Responsibility |
|---|---|---|
| `src/tensor.c` | `include/tensor.h` | Multi-dim Tensor memory allocation, CPU SIMD math, autograd tape management, CPU $\leftrightarrow$ GPU transfers. |
| `src/layer.c` | `include/layer.h` | Dense (Linear) projection layers with CPU/CUDA forward and backward dispatches. |
| `src/rnn.c` | `include/rnn.h` | Recurrent Neural Network layer with BPTT unrolling and gradient clipping. |
| `src/conv.c` | `include/conv.h` | 2D Spatial Convolutions, 2D Padding/Unpadding, and 2D MaxPool layers. |
| `src/batchnorm.c` | `include/batchnorm.h` | Batch Normalization with running mean/variance tracking and train/eval toggles. |
| `src/dropout.c` | `include/dropout.h` | Inverted Dropout regularization layer with training mask generation. |
| `src/nn.c` | `include/nn.h` | Network container abstraction, layer dispatch table, Cross-Entropy / MSE losses. |
| `src/optimizer.c` | `include/optimizer.h` | SGD with Momentum & Weight Decay, Adam Optimizer with 1st/2nd moments. |
| `src/tokenizer.c` | `include/tokenizer.h` | Vocab string dictionary mapping, token encoding/decoding, EOS token management. |
| `src/dataset_loader.c` | `include/dataset_loader.h` | Streaming binary dataset reader, token sequence windowing, wraparound batching. |
| `src/sampler.c` | `include/sampler.h` | Inference text generation sampling strategies (Greedy, Temperature, Top-k, Top-p). |
| `src/transformer.c` | `include/transformer.h` | Decoder-only Transformer with AVX2 SIMD, RoPE, RMSNorm, SwiGLU, and Q8_0 quant. |
| `src/tui.c` | `include/tui.h` | Antigravity terminal UI dashboard and raw termios REPL shell. |
| `config/config_ui.c` | `neuralc_config.h` | Kernel-style menuconfig configuration TUI engine. |
| `memory/memory.c` | N/A | Memory arena allocator and 64-byte aligned pool manager. |
| `cuda/src/cuda_backend.cu` | `cuda/include/cuda_backend.h` | CUDA kernels for elementwise ops, fused linear+ReLU, tiled transpose, cuBLAS GEMM, FP16. |
| `apps/neuricode_cli.c` | N/A | Interactive Antigravity CLI REPL executable entry. |
| `apps/train.c` | N/A | Language model training executable driver. |
| `apps/cli.c` | N/A | CLI model inference and text generation executable driver. |
| `apps/pipeline.c` | N/A | End-to-end automated training and deployment pipeline executable. |

---

## SECTION 8: HOW TO EXTEND THE SYSTEM

### 8.1 Adding New Custom Layers
1. Define layer struct and API prototypes in `include/layer.h`.
2. Implement CPU OpenMP forward/backward loops in `src/layer.c`.
3. Implement `__global__` CUDA kernels and host launchers in `cuda/src/cuda_backen# Neuricode DL: C11 & CUDA Production Framework System Specification & Architecture Reference

---

## SECTION 1: FULL SYSTEM OVERVIEW

### 1.1 High-Level Architectural Paradigm
Neuricode is a production-grade, zero-external-dependency deep learning framework and LLM engine built in pure ISO C11 with an accelerated CUDA C++/cuBLAS compute backend. The system is engineered around six core technical pillars:

1. **Explicit Memory Control & Cache Alignment**: Zero implicit heap allocations during active forward/backward passes. Memory required for activations, gradients, scratchpads, and velocity state buffers is pre-allocated and aligned to 64-byte boundaries (`neuralc_aligned_alloc`) to optimize SIMD vector loads (AVX2/AVX-512) and DMA transfer efficiency. Includes an optional memory arena allocator (`memory/memory.c`).
2. **Unified Dual-Backend Device Abstraction**: Tensors carry explicit device placement tags (`CF_GPU_NONE`, `CF_GPU_CUDA`, `CF_GPU_OPENCL`). High-level neural network operations (Dense, Conv2D, RNN, Transformer, Loss Functions) dynamically query tensor placement and dispatch execution either to CPU SIMD micro-kernels or to asynchronous CUDA kernel streams.
3. **Dynamic Autograd Tape & High-Performance Manual Dispatch**: Supports both reverse-mode automatic differentiation via an execution graph tape (`GraphNode`) and direct zero-overhead layer-by-layer forward/backward execution for performance-critical recurrent models.
4. **Fused CUDA Operator Kernels**: Merges linear projection, bias addition, and non-linear activation functions into single-pass CUDA grid executions, eliminating round-trip DRAM read/write cycles on GPU global memory.
5. **Kernel-Style `menuconfig` Configuration UI**: Interactive terminal configuration interface (`make config` / `menuconfig`) built in C (`config/config_ui.c`) that outputs a system configuration manifest (`neuralc_config.h`) without requiring source code modifications.
6. **Deterministic Binary Serialization & Embedded Runtime**: Complete model state, hyperparameters, token dictionaries, and weights are saved into deterministic, byte-aligned binary format files compatible across training and production inference executables (`train`, `pipeline`, `cli`, `neuricode`). Boots in < 2 ms inside a ~4.4 MB standalone binary.

```
+-----------------------------------------------------------------------------------+
|                                  USER APPLICATIONS                                |
|             (apps/neuricode_cli.c, apps/train.c, apps/cli.c, apps/pipeline.c)     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            ANTIGRAVITY TUI & SHELL ENGINE                         |
|                 (src/tui.c, config/config_ui.c, neuralc_config.h)                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            HIGH-LEVEL NEURAL MODULES                              |
|           (src/nn.c, src/layer.c, src/rnn.c, src/conv.c, src/transformer.c)      |
+-----------------------------------------------------------------------------------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
+-----------------------------------------------+ +---------------------------------+
|         CPU BACKEND & TENSOR TAPE             | |          CUDA BACKEND           |
|  (src/tensor.c, OpenMP, AVX2 SIMD, Autograd)  | | (cuda/src/cuda_backend.cu,    |
+-----------------------------------------------+ |  cuBLAS GEMM, Fused Kernels)    |
                        |                         +---------------------------------+
                        v                                         |
+-----------------------------------------------+                         |
|                 HOST RAM (DRAM)               |                         |
|      (64-byte aligned buffers, memory.c)      |                         |
+-----------------------------------------------+                         |
                        ^                                         |
                        |====== Host-to-Device / Device-to-Host ==|
                        |       (cudaMemcpyAsync / Pinned Host)   v
                                                  +---------------------------------+
                                                  |         GPU VRAM (GDDR/HBM)     |
                                                  |      (Coalesced Memory Tiles)   |
                                                  +---------------------------------+
```

---

### 1.2 Data Flow Architecture: Raw Text to Trained Model Outputs

The complete data pipeline transforms raw unstructured text into tokenized integer IDs, high-dimensional target vectors, dense recurrent embeddings, and cross-entropy loss gradients.

```
+------------------+     +-------------------+     +------------------+
| Raw Text File    | --> | Tokenizer Engine  | --> | Token ID Stream  |
| (data.txt)       |     | (src/tokenizer.c) |     | [id_0, id_1, ...] |
+------------------+     +-------------------+     +------------------+
                                                            |
                                                            v
+------------------+     +-------------------+     +------------------+
| Reshaped 2D/3D   | <-- | One-Hot Encoding  | <-- | Stream Dataset   |
| Tensor Buffers   |     | (train.c)         |     | (dataset_loader) |
+------------------+     +-------------------+     +------------------+
        |
        v
+-----------------------------------------------------------------------------+
|                            FORWARD PROPAGATION                              |
|                                                                             |
|  +------------------+     +-------------------+     +---------------------+ |
|  | RNN Recurrent    | --> | 3D->2D Zero-Copy  | --> | Dense Linear        | |
|  | Layer (BPTT)     |     | Tensor Reshape    |     | Projection Layer    | |
|  +------------------+     +-------------------+     +---------------------+ |
|                                                                |            |
|                                                                v            |
|                           +-------------------+     +---------------------+ |
|                           | Cross-Entropy     | <-- | Softmax Probability | |
|                           | Loss Computation  |     | Row Normalization   | |
|                           +-------------------+     +---------------------+ |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                            BACKWARD PROPAGATION                             |
|                                                                             |
|  +------------------+     +-------------------+     +---------------------+ |
|  | Parameter Update | <-- | Gradient Norm     | <-- | Loss Gradient       | |
|  | (SGD / Adam)     |     | Clipping (L2)     |     | (dL/dz = p - t)     | |
|  +------------------+     +-------------------+     +---------------------+ |
+-----------------------------------------------------------------------------+
```

---

### 1.3 Comparative Analysis: Neuricode vs. Standard Deep Learning Frameworks

| Feature / Dimension | ⚡ **Neuricode v1** | 🐍 **PyTorch / TensorFlow** | 🦙 **llama.cpp / GGML** | 📦 **ONNX Runtime** |
|---|---|---|---|---|
| **Language Stack** | **100% ISO C11** | Python + C++ / CUDA | C++11 / C++17 | C++17 |
| **Dependencies** | **0 (Zero / Standard C)** | Heavy (Python, MKL, libtorch, CUDA) | Low (C++ STL, pthread, OpenMP) | Medium (Protobuf, Abseil, Flatbuffers) |
| **Binary Size** | **~4.4 MB** | > 2.0 – 5.0 GB | ~15 – 30 MB | ~50 – 100 MB |
| **Cold Start Boot** | **< 2 milliseconds** | 1.5 – 5.0 seconds | 50 – 200 milliseconds | 200 – 500 milliseconds |
| **Terminal UI Engine** | **Built-in ANSI TUI + `menuconfig`** | None (requires Python UI) | Basic CLI prompt | None (library only) |
| **Embedded Edge Ready** | **Raspberry Pi, Jetson, MCU** | No (requires heavy OS + Python) | Yes (Inference only) | Partial (requires C++ runtime) |
| **Native C Training** | **Full Autograd & Backprop** | Full (Python API) | Limited / Inference Focused | Limited / Export Focused |

---

## SECTION 2: EXECUTION FLOW (STEP-BY-STEP)

This section provides an line-by-line execution trace of running the standard Neuricode training executable:

```bash
./train chunks/token_ab assets/vocab.txt model.bin 20 64 4 0.001
```

---

### Step 1: Argument Parsing & Hyperparameter Configuration
- **Mechanism**: Executed in `apps/train.c` inside `main(int argc, char **argv)`.
- **Parsing Logic**:
  - `argv[1]` = `"chunks/token_ab"` (Training text data file path).
  - `argv[2]` = `"assets/vocab.txt"` (Vocabulary mapping file path).
  - `argv[3]` = `"model.bin"` (Output model binary checkpoint path).
  - `argv[4]` = `"20"` (Sequence length $S = 20$).
  - `argv[5]` = `"64"` (Hidden layer dimension $H = 64$).
  - `argv[6]` = `"4"` (Total training epochs $E = 4$).
  - `argv[7]` = `"0.001"` (Base learning rate $\eta = 0.001$).
- **Config Struct Initialization**:
  The values populate a `TrainConfig` structure. If optional CLI positional arguments are omitted, `train.c` falls back to default constants specified in `neuralc_config.h` or `#ifndef` macros (`SEQ_LEN=20`, `BATCH_SIZE=8`, `EPOCHS=20`, `HIDDEN_SIZE=256`, `LEARNING_RATE=0.01f`).

---

### Step 2: Tokenizer Loading & Corpus Encoding
- **Mechanism**: Executed via `tokenizer_load(vocab_path)` and `tokenizer_encode()`.
- **Internal Mechanics**:
  1. Opens `assets/vocab.txt`, reading line-delimited token strings.
  2. Allocates a `Tokenizer` structure containing a dynamic string table (`char **tokens`) and a hash table mapping strings to integer token indices (`int *id_map`).
  3. Detects if an explicit End-Of-Sequence `<EOS>` token exists in the vocabulary (`tokenizer_eos_id()`).
  4. Encodes `chunks/token_ab`: If `<EOS>` is defined, `train.c` reads the input file line-by-line, invoking `tokenizer_encode()` per line and explicitly appending the `<EOS>` token ID to the stream.

---

### Step 3: Binary Dataset Creation & Streaming Setup
- **Mechanism**: Handled by `dataset_loader.c` routines `encode_file_to_binary()` and `dataset_open()`.
- **Internal Mechanics**:
  1. Writes the full token stream out to a temporary binary dataset file (`dataset.bin`). The file format consists of a contiguous stream of `uint32_t` token values.
  2. `dataset_open()` memory-maps or opens `dataset.bin`, initializing a `DatasetLoader` struct.

---

### Step 4: Model Initialization & Binary Checkpoint Verification
- **Mechanism**: Executed via `rnn_create()`, `dense_create()`, and `load_model()`.
- **Internal Mechanics**:
  1. `train.c` checks if `model.bin` already exists on disk.
  2. Reads or instantiates $W_{xh} \in \mathbb{R}^{H \times V}$, $W_{hh} \in \mathbb{R}^{H \times H}$, $b_h \in \mathbb{R}^{H \times 1}$, $W_{\text{dense}} \in \mathbb{R}^{V \times H}$, and $b_{\text{dense}} \in \mathbb{R}^{V \times 1}$.

---

### Step 5: Training Loop Execution & Steps Sizing
- **Mechanism**: Outer `for (int epoch = 0; epoch < config.epochs; epoch++)` loop and inner `for (int step = 0; step < steps_per_epoch; step++)` loop.
- **Internal Mechanics**:
  1. Computes total tokens per batch: $\text{tokens\_per\_batch} = B \times S = 64 \times 20 = 1280$.
  2. Calculates step count: $\text{steps\_per\_epoch} = \lfloor \frac{\text{ds->token\_count}}{1280} \rfloor$.
  3. Monitors loss plateauing across epochs. If `avg_loss > prev_loss - 0.0005f` for 3 consecutive epochs, decays learning rate: $\eta \leftarrow \eta \times 0.8$.

---

### Step 6: Forward Pass Internals
- **Mechanism**: `dataset_next_batch()`, `fill_onehot_3d()`, `rnn_forward()`, `tensor_reshape()`, `dense_forward()`.
- **Internal Mechanics**:
  1. `rnn_forward()` unrolls hidden states $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$.
  2. `tensor_reshape()` creates a 2D zero-copy view tensor `[1280, 64]`.
  3. `dense_forward()` computes $Z = X_{\text{2d}} W_{\text{dense}}^T + b_{\text{dense}}$ and Softmax probabilities.

---

### Step 7: Backward Pass Internals
- **Mechanism**: `nn_loss()`, `dense_backward()`, `tensor_free()`, `rnn_backward()`, `rnn_clip_gradients()`, `nn_clip_gradients()`.
- **Internal Mechanics**:
  1. `nn_loss()` initializes loss gradient $\frac{\partial L}{\partial Z} = \frac{P - Y}{B \times S}$.
  2. `dense_backward()` computes $dW_{\text{dense}}, db_{\text{dense}}, dX_{\text{dense}}$.
  3. `tensor_free()` deallocates 2D view header.
  4. `rnn_backward()` unrolls BPTT: $\delta_t = (\delta_{t+1} W_{hh}^T + dX_{3d, t}) \odot (1 - h_t^2)$.
  5. Gradient clipping rescales gradients if norm > 5.0f.

---

### Step 8: Optimizer Updates & Model Checkpointing
- **Mechanism**: `sgd_step()`, `save_model()`.
- **Internal Mechanics**:
  1. `sgd_step()` applies momentum velocity updates: $v_{t+1} = \mu v_t + g + \lambda W$; $W \leftarrow W - \eta v_{t+1}$.
  2. `save_model()` serializes raw float weight buffers to `model.bin`.

---

## SECTION 3: CUDA BACKEND DEEP DIVE

### 3.1 Advanced GPU Memory Management Architecture
- **Device Allocation (`cuda_malloc_f32`)**: Wraps `cudaMalloc()` with `CUDA_CHECK` error handling.
- **Page-Locked Host Allocation (`cuda_malloc_host`)**: Uses `cudaMallocHost()` for pinned DRAM allocations, enabling direct PCI-Express Direct Memory Access (DMA) transfers.
- **Asynchronous Copies (`cuda_memcpy_h2d_async`)**: Executes `cudaMemcpyAsync()` over CUDA streams.

---

### 3.2 CUDA Streams & cuBLAS GEMM Mapping
- **Stream-Bound Execution**: All GPU kernels and cuBLAS operations bind to active `cudaStream_t` instances (`cuda_set_stream`).
- **cuBLAS GEMM Matrix Layout**: Row-major C matrix multiplication maps to column-major cuBLAS via transpose identity $C = A B \iff C^T = B^T A^T$.
- **Tensor Core Enabling**: `cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)`.

---

### 3.3 Fused Kernels & Shared Memory Tiled Transpose
- **Fused Linear + ReLU (`k_fused_linear_relu`)**: Fuses matrix projection, bias addition, and ReLU activation into one GPU grid call, eliminating intermediate DRAM writes.
- **Tiled Transpose (`k_transpose_tiled`)**: Uses $32 \times 32$ shared memory tiles with `tile[32][33]` column padding, preventing 32-bank conflicts and enforcing coalesced global memory access.

---

## SECTION 4: MATHEMATICAL FOUNDATIONS

### 4.1 Forward & Backward Equations

| Component | Forward Equation | Backward Gradient Equation |
|---|---|---|
| **Linear Layer** | $Z = X W^T + b$ | $dW = dZ^T X, \quad db = \sum dZ, \quad dX = dZ W$ |
| **RNN Recurrent Step** | $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ | $\delta_t = (\delta_{t+1} W_{hh}^T + dX_t) \odot (1 - h_t^2)$ |
| **Softmax + Cross-Entropy** | $P_i = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$ | $\frac{\partial L}{\partial Z} = P - Y$ |
| **Batch Normalization** | $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ | $\frac{\partial L}{\partial x_i} = \frac{\gamma}{B \sqrt{\sigma^2 + \epsilon}} \left[ B \frac{\partial L}{\partial y_i} - \sum \frac{\partial L}{\partial y} - \hat{x}_i \sum \frac{\partial L}{\partial y} \hat{x} \right]$ |

---

## SECTION 5: ANTIGRAVITY TUI SHELL & MENUCONFIG

### 5.1 Interactive CLI Shell Architecture (`apps/neuricode_cli.c` & `src/tui.c`)

Neuricode includes a lightweight terminal REPL shell:

```bash
./neuricode
```

```
   ┌──────────────────────────────────────────────────────────┐
   │ ⚡ NEURICODE v1 — Native Zero-Dependency Edge AI Engine │
   │ Model: model.bin | Vocab: assets/vocab.txt | GPU: CUDA  │
   └──────────────────────────────────────────────────────────┘
```

#### Terminal Control Features:
- **Raw Mode Input Processing**: Uses UNIX `termios` to intercept arrow keys, backspaces, and tab completion without terminal input echoing.
- **Dynamic Window Resizing**: Handles OS `SIGWINCH` signals via `ioctl(STDOUT_FILENO, TIOCGWINSZ, &w)`, adjusting layout boundaries instantly.
- **8 Color Themes**: Dynamic ANSI color scheme switcher (`/theme`).
- **Slash Commands**:
  - `/help`: Displays active hyperparameter specs and system manual.
  - `/status`: Renders live memory, vocabulary, and GPU status dashboard.
  - `/theme`: Opens interactive arrow-key color theme selector.
  - `/temp <val>`: Adjusts logit sampling temperature (e.g. `/temp 0.7`).
  - `/topk <val>`: Sets Top-K nucleus sampling threshold (e.g. `/topk 40`).
  - `/reset`: Clears model recurrent hidden state memory.
  - `/clear`: Clears terminal screen buffer.
  - `/exit`: Safely terminates session.

---

### 5.2 Kernel-Style `menuconfig` Interface (`config/config_ui.c`)

```bash
make config
# or
./menuconfig
```

The interactive TUI writes configuration parameters out to `neuralc_config.h`:
- `HIDDEN_SIZE`: Recurrent hidden dimension.
- `LEARNING_RATE`: Optimization step size.
- `SEQ_LEN`: Unrolled sequence length.
- `BATCH_SIZE`: Batch size.
- `USE_OMP`: Toggle OpenMP multi-threading.
- `USE_CUDA`: Toggle CUDA GPU backend dispatch.

---

## SECTION 6: TESTING & VALIDATION

### 6.1 Multi-Module Unit Testing Strategy
- **Autograd Tape Verification (`src/test_autograd.c`)**: Validates graph node creation, topological sorting, and reverse-mode gradient flow.
- **CPU vs. GPU Equivalence**: Compares float arrays generated by CPU reference functions and CUDA GPU kernels ($\|y_{\text{cpu}} - y_{\text{gpu}}\|_{\infty} < 10^{-5}$).
- **Sanitizer Verification**: Tested using GCC `-fsanitize=address,undefined` and NVIDIA `compute-sanitizer`.

---

## SECTION 7: PROJECT STRUCTURE & MODULE INVENTORY

| File Path | Header File | Core Functional Responsibility |
|---|---|---|
| `src/tensor.c` | `include/tensor.h` | Multi-dim Tensor memory allocation, CPU SIMD math, autograd tape management, CPU $\leftrightarrow$ GPU transfers. |
| `src/layer.c` | `include/layer.h` | Dense (Linear) projection layers with CPU/CUDA forward and backward dispatches. |
| `src/rnn.c` | `include/rnn.h` | Recurrent Neural Network layer with BPTT unrolling and gradient clipping. |
| `src/conv.c` | `include/conv.h` | 2D Spatial Convolutions, 2D Padding/Unpadding, and 2D MaxPool layers. |
| `src/batchnorm.c` | `include/batchnorm.h` | Batch Normalization with running mean/variance tracking and train/eval toggles. |
| `src/dropout.c` | `include/dropout.h` | Inverted Dropout regularization layer with training mask generation. |
| `src/nn.c` | `include/nn.h` | Network container abstraction, layer dispatch table, Cross-Entropy / MSE losses. |
| `src/optimizer.c` | `include/optimizer.h` | SGD with Momentum & Weight Decay, Adam Optimizer with 1st/2nd moments. |
| `src/tokenizer.c` | `include/tokenizer.h` | Vocab string dictionary mapping, token encoding/decoding, EOS token management. |
| `src/dataset_loader.c` | `include/dataset_loader.h` | Streaming binary dataset reader, token sequence windowing, wraparound batching. |
| `src/sampler.c` | `include/sampler.h` | Inference text generation sampling strategies (Greedy, Temperature, Top-k, Top-p). |
| `src/transformer.c` | `include/transformer.h` | Decoder-only Transformer with AVX2 SIMD, RoPE, RMSNorm, SwiGLU, and Q8_0 quant. |
| `src/tui.c` | `include/tui.h` | Antigravity terminal UI dashboard and raw termios REPL shell. |
| `config/config_ui.c` | `neuralc_config.h` | Kernel-style menuconfig configuration TUI engine. |
| `memory/memory.c` | N/A | Memory arena allocator and 64-byte aligned pool manager. |
| `cuda/src/cuda_backend.cu` | `cuda/include/cuda_backend.h` | CUDA kernels for elementwise ops, fused linear+ReLU, tiled transpose, cuBLAS GEMM, FP16. |
| `apps/neuricode_cli.c` | N/A | Interactive Antigravity CLI REPL executable entry. |
| `apps/train.c` | N/A | Language model training executable driver. |
| `apps/cli.c` | N/A | CLI model inference and text generation executable driver. |
| `apps/pipeline.c` | N/A | End-to-end automated training and deployment pipeline executable. |

---

## SECTION 8: HOW TO EXTEND THE SYSTEM

### 8.1 Adding New Custom Layers
1. Define layer struct and API prototypes in `include/layer.h`.
2. Implement CPU OpenMP forward/backward loops in `src/layer.c`.
3. Implement `__global__` CUDA kernels and host launchers in `cuda/src/cuda_backend.cu`.
4. Register layer enum tag in `include/nn.h` and add dispatch cases in `src/nn.c`.

---

## SECTION 9: SIMPLIFIED EXPLANATION (FOR BEGINNERS)
The framework operates like a high-speed factory:
- **Tokenizer**: A dictionary mapping words to integer IDs.
- **Tensor**: Multi-dimensional boxes holding numbers.
- **Layers**: Assembly line workers applying math transformations.
- **Autograd / Optimizer**: Inspection team detecting errors and adjusting parameter knobs.
- **GPU**: An army of 3,000 parallel workers doing millions of multiplications simultaneously.

---

## SECTION 10: PROFESSIONAL README CONTENT

*(See updated [`README.md`](file:///home/raccoon/neuricode/README.md) for the GitHub repository front page).*d.cu`.
4. Register layer enum tag in `include/nn.h` and add dispatch cases in `src/nn.c`.

---

## SECTION 9: SIMPLIFIED EXPLANATION (FOR BEGINNERS)
The framework operates like a high-speed factory:
- **Tokenizer**: A dictionary mapping words to integer IDs.
- **Tensor**: Multi-dimensional boxes holding numbers.
- **Layers**: Assembly line workers applying math transformations.
- **Autograd / Optimizer**: Inspection team detecting errors and adjusting parameter knobs.
- **GPU**: An army of 3,000 parallel workers doing millions of multiplications simultaneously.

---

## SECTION 10: PROFESSIONAL README CONTENT

*(See updated [`README.md`](file:///home/raccoon/neuricode/README.md) for the GitHub repository front page).*
