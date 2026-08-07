#  Neuricode — Native Zero-Dependency Edge AI Engine & C11 Deep Learning Framework

<p align="center">
  <b>A lightweight, high-performance Deep Learning Framework & Transformer / LLM Engine written from first principles in 100% Pure ISO C11.</b>
</p>

---

> [!IMPORTANT]
> **What is Neuricode ? Is it a Deep Learning Framework or an LLM Engine?**  
> **It is BOTH!**
> 1. **A Native C11 Deep Learning Framework**: Implements $N$-dimensional tensors, automatic differentiation (autograd), matrix GEMM, layers (Linear, Conv2D, BatchNorm, Dropout), optimizers (SGD, Adam), and high-performance CUDA GPU backends.
> 2. **An Embedded Generative AI / LLM & Sequence Engine**: Built directly on top of the framework, executing multi-head attention Transformers, Recurrent Neural Networks (RNNs), byte/word tokenization, and dynamic Top-K / Top-P nucleus sampling in a tiny **~4.4 MB standalone binary**.

---

## Creator's Vision & Why Neuricode is Unique

Modern Artificial Intelligence is heavily reliant on massive 2–5 GB Python frameworks (PyTorch, TensorFlow) requiring Python interpreters, CUDA toolchains, and expensive GPUs. Conversely, C++ inference engines (like `llama.cpp` or `ONNX Runtime`) focus strictly on executing pre-trained models and require C++ runtimes and complex build pipelines.

**Neuricode v1 was engineered to bridge this gap.** Built entirely in standard C (C11/POSIX) with **zero external library dependencies**, Neuricode v1 provides a unified engine for both training (backpropagation) and generative inference. It boots in under **2 milliseconds** and runs natively on resource-constrained embedded platforms—including **Raspberry Pi, NVIDIA Jetson, microcontrollers, edge IoT devices, and Linux servers**.

---

##  Key Highlights & Engineering Innovations

- **Zero Third-Party Dependencies**: Written in 100% ISO C11. No PyTorch, no Python, no BLAS/LAPACK, no OpenBLAS/MKL requirement.
- **Edge & Embedded System Ready**: Boots instantly (< 2 ms) with a tiny memory footprint (~4.4 MB binary).
- **Full Autograd & Training Engine**: Reverse-mode automatic differentiation, backpropagation through time (BPTT), SGD with momentum, Adam optimizer, gradient clipping, and checkpoint serialization.
- **Native Transformer / LLM Architecture**: Multi-head self-attention, scaled dot-product attention, positional embeddings, RMSNorm/LayerNorm, byte/word tokenizers, and Temperature/Top-K/Top-P samplers.
- **Asynchronous CUDA Stream GPU Backend**: Stream-bound cuBLAS (`cublasSetStream`), pinned host memory (`cudaMallocHost`), non-blocking transfers (`cudaMemcpyAsync`), $32 \times 32$ tiled shared-memory matrix transpose (zero bank conflicts), fused kernels (`linear_relu`, `add_relu`), multi-GPU device selection, and FP16 Tensor Core support.
- **Linux Kernel-Style `menuconfig` TUI**: Interactive terminal configuration interface (`make config`) to adjust hyperparameters and hardware backends without modifying source code.
- **Antigravity CLI Shell**: Interactive terminal prompt with real-time `SIGWINCH` window resizing (`ioctl`), 8 selectable color themes, double-buffered rendering, and live system status dashboards.

---

##  System Architecture Overview

Neuricode is designed as a decoupled, modular architecture:

```text
                                  ┌─────────────────────────────────────────┐
                                  │      Neuricode CLI Shell (tui.c)        │
                                  └────────────────────┬────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │    Inference & Step Pipeline            │
                                  │  (pipeline.c, sampler.c, tokenizer.c)   │
                                  └────────────────────┬────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │    Neural Architecture & Layers         │
                                  │    (transformer.c, rnn.c, layer.c)      │
                                  └────────────────────┬────────────────────┘
                                                       │
                                  ┌────────────────────▼────────────────────┐
                                  │    Core Tensor & Autograd Engine        │
                                  │       (tensor.c, optimizer.c)           │
                                  └──────────┬───────────────────┬──────────┘
                                             │                   │
                     ┌───────────────────────▼──────┐     ┌──────▼──────────────────────┐
                     │ OpenMP Threading (CPU Acceleration) │     │ CUDA Backend (Stream/SMEM GPU)│
                     └──────────────────────────────┘     └─────────────────────────────┘
```

### Directory Structure

```text
neuricode/
├── apps/                  # CLI & Application Entry Points
│   ├── neuricode_cli.c    # Interactive Antigravity CLI Shell & REPL loop
│   ├── train.c            # Streaming dataset model trainer
│   └── cli.c              # Command-line inference utility
├── cuda/                  # High-Performance CUDA GPU Backend
│   ├── include/
│   │   └── cuda_backend.h # Primary C-API CUDA backend header
│   └── src/
│       └── cuda_backend.cu# Stream-bound CUDA kernels & tiled SMEM transpose
├── include/               # Public Engine C Headers
│   ├── tensor.h           # N-dimensional tensor math & autograd header
│   ├── layer.h            # Neural layer definitions (Linear, Conv2D, Softmax)
│   ├── transformer.h      # Multi-head attention Transformer header
│   ├── pipeline.h         # Model loader & sequence step pipeline
│   ├── tokenizer.h        # Byte/word tokenizer header
│   └── tui.h              # Antigravity Terminal UI engine header
├── src/                   # Core Engine Implementations
│   ├── tensor.c           # Contiguous memory array math & vectorization
│   ├── transformer.c      # Transformer forward & self-attention math
│   ├── rnn.c              # Recurrent Neural Network layers & BPTT
│   ├── layer.c            # Dense layers, activations, loss functions
│   ├── optimizer.c        # SGD & Adam optimizer algorithms
│   ├── tokenizer.c        # Greedy vocabulary encoder & decoder
│   ├── sampler.c          # Logit temperature & Top-K / Top-P samplers
│   └── tui.c              # Raw-mode termios & SIGWINCH layout engine
├── config/                # Terminal Configuration UI
│   ├── config_ui.c        # Kernel-style menuconfig TUI engine
│   └── neuralc_config_main.c # Standalone config executable entry
├── memory/                # Memory Arena & Allocator
│   └── memory.c           # Memory pool allocator
├── neuralc_config.h       # System Configuration Manifest (Generated by menuconfig)
└── makefile               # Pure C11 build automation & auto-config loader
```


##  Quickstart & Building Guide

### 1. Prerequisite
Standard C compiler (`gcc` or `clang`), `make`, and optional NVIDIA CUDA Toolkit (`nvcc`) for GPU acceleration.

### 2. Build Neuricode CLI
Compile the production binary in one command:

```bash
make neuricode
```

To install system-wide to `~/.local/bin/neuricode`:

```bash
make install
```

### 3. Interactive Hyperparameter Configuration (`menuconfig`)
Adjust parameters, OpenMP threading, and CUDA GPU backends via Linux kernel-style TUI:

```bash
make config
```

### 4. Run the Interactive Terminal Shell
Launch the interactive AI shell:

```bash
neuricode
```

Inside `neuricode`, use `/` commands to control execution:

| Command | Description |
|---|---|
| `/help` | Display manual, active model specs, and hyperparameters |
| `/status` | View system dashboard (Model, Vocab, Hardware status) |
| `/theme` | Open interactive arrow-key color theme selector (8 themes) |
| `/temp <val>` | Set sampling temperature dynamically (e.g. `/temp 0.30`) |
| `/topk <val>` | Set Top-K sampling cap dynamically (e.g. `/topk 10`) |
| `/reset` | Reset model hidden state memory |
| `/clear` | Clear terminal screen |
| `/exit` | Exit Neuricode CLI |

---

## 📜 License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
