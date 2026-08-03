# Neuricode — Native Zero-Dependency Edge AI Engine in Pure C. A similar version of Neurix.


##  Key Highlights & Engineering Features

- **Zero External Dependencies**: 100% written in standard C (C11/POSIX). No PyTorch, no Python, no heavy third-party math bloat.
- **Embedded & Edge Systems Ready**: Engineered for ultra-low footprint devices. Runs anywhere standard C compiles.
- **Classic Antigravity CLI Interface**: Sleek terminal UI featuring straight divider frames, purple aesthetics, 8 selectable color themes, and smooth character typing animations.
- **Dynamic Window Auto-Resizing (`SIGWINCH`)**: Catches OS window resize signals (`ioctl`) to automatically adjust border frames live when minimizing, maximizing, or dragging your terminal window.
- **Dynamic Path Auto-Discovery**: Automatically locates model weights (`model.bin`) and vocabularies (`assets/vocab.txt`) across environment variables, working directories, or binary location (`/proc/self/exe`).
- **Kernel-Style `menuconfig` TUI**: Linux kernel-inspired terminal interface (`make config`) to adjust hyperparameters without touching source code.
- **Multi-Core Hardware Acceleration**: Parallel execution via OpenMP CPU threading with optional CUDA & OpenCL GPU backend hooks.
- **Added Decoder-Only Transformer Engine**.
    1. SIMD-Accelerated (AVX2 / ARM NEON / Unrolled Scalar) Tiled Matrix Multiplication
    2. Precomputed RoPE (Rotary Position Embeddings) Cos/Sin Lookup Tables
    3. Optimized KV-Cache Layout for Contiguous Sequential Memory Prefetching
    4. Strict 64-Byte Aligned Memory Allocations via neuralc_aligned_alloc
    5. Numerically Stable Softmax and RMSNorm (Log-Sum-Exp / Epsilon / NaN Guards)
    6. Debug Mode & Tensor Validation (NaN / Inf Diagnostics)
    7. Pretrained Binary Weight Loading & Saving API
    8. Multi-Token Batch Forward Inference Engine
    9. INT8 (Q8_0) Quantization Support for Ultra-Low Footprint Inference

  - Support HIGH-END / LOW END MICROKERNEL'S
  - Added Feature in control to change the RNN / Transformer Via TUI and,
  - Added in Config menu "Transformer Engine Setup"
      - Vocabulary size
      - Hidden Dimension
      - FFN Expansion Dim
      - Number of LAyers
      - Number of Heads
      - Max context Length




---

##  How Neuricode Works (Under the Hood)

For new users wanting to understand the inner workings of Neuricode v1, the architecture is divided into decoupled, modular C translation units:

```text
Neuricode/
├── apps/                  # Application Entry Points
│   ├── neuricode_cli.c    # Antigravity interactive CLI shell & REPL loop
│   ├── train.c            # Streaming dataset model trainer
│   ├── cli.c              # Basic inference utility
│   └── pipeline.c         # High-level model loader & inference pipeline
├── include/               # Public C Headers
│   ├── tui.h              # Antigravity TUI engine header
│   ├── tensor.h           # N-dimensional tensor math header
│   ├── pipeline.h         # Model loader & step pipeline header
│   └── tokenizer.h        # Tokenizer & vocabulary header
├── src/                   # Core Math & Engine Implementation
│   ├── tui.c              # Terminal UI engine (termios raw mode, SIGWINCH, themes)
│   ├── tensor.c           # Custom row-major matrix/tensor memory routines
│   ├── rnn.c              # Recurrent Neural Network layers & BPTT algorithms
│   ├── layer.c            # Dense (linear) layers, Softmax, activations
│   ├── optimizer.c        # SGD with momentum, weight decay, gradient clipping
│   ├── tokenizer.c        # Greedy byte/word vocabulary encoder & decoder
│   ├── sampler.c          # Temperature, Top-K, Top-P (Nucleus) samplers
│   └── dataset_loader.c   # Streaming binary batch dataset loader
└── makefile               # C11 build automation & auto-config loader
```

### The Execution Steps:
1. **Tensor Math (`src/tensor.c`)**: Manages contiguous N-dimensional floating-point arrays, matrix multiplications (`GEMM`), vector additions, and SIMD vectorizations.
2. **Neural Forward Pass & BPTT (`src/rnn.c`, `src/layer.c`)**: Calculates hidden state transitions across timesteps for sequence generation and backpropagates gradients through time during training.
3. **Logit Sampling (`src/sampler.c`)**: Scaled Softmax probabilities using Temperature and Top-K filtering to select next-token output IDs.
4. **Antigravity TUI Shell (`src/tui.c`, `apps/neuricode_cli.c`)**: Renders framed terminal prompts, handles Linux raw-mode keyboard inputs, listens to `SIGWINCH` resize events, and streams generated text output smoothly.

---

## Quickstart Guide for New Users

### 1. Build the Neuricode Assistant
Compile the native binary with zero external dependencies and install to system PATH:

```bash
make neuricode
```
```bash
make install
```

### 2. Run the Interactive CLI
Launch the terminal UI directly from anywhere in your shell by simply typing:

```bash
neuricode
```

Or run locally within the repository directory:

```bash
./neuricode
```

### 3. Interactive Slash Commands Inside CLI
Inside `neuricode`, you can type commands starting with `/`:

| Command | Action |
|---|---|
| `/help` | Display command manual and active hyperparameters |
| `/theme` | Open interactive arrow-key selector (8 Color Palettes) |
| `/status` | Display system dashboard (Model, Vocab, Hardware status) |
| `/temp <val>` | Set sampling temperature dynamically (e.g. `/temp 0.30` for coherent text) |
| `/topk <val>` | Set Top-K sampling cap dynamically (e.g. `/topk 10`) |
| `/log` | Run test demonstration of colored log badges |
| `/reset` | Reset model hidden state memory |
| `/clear` | Clear terminal screen |
| `/exit` | Quit Neuricode CLI |

---

## 8 Selectable Color Themes

Type `/theme` in the CLI to switch color palettes using **↑ / ↓ Arrow Keys**:

1. **Classic Purple (Default)** — Sleek Purple / Magenta / White
2. **Cyberpunk Cyan** — Neon Cyan / Pink
3. **Matrix Green** — Electric Green / Yellow
4. **Sunset Orange** — Neon Orange / Gold
5. **Electric Blue** — Cobalt Blue / Bright Cyan
6. **Crimson Red** — Neon Crimson / Coral
7. **Emerald Mint** — Teal / Mint Green
8. **Monochrome Dark** — Slate Gray / Silver Minimalist

---

## Training Your Own Model on Custom Text

Want to train Neuricode on your own text dataset (books, code, or stories)?

### Step 1: Configure Hyperparameters
Launch the kernel-style configuration menu:

```bash
make config
```

Use Arrow Keys to set `HIDDEN_SIZE` (e.g. 256 or 512) and `EPOCHS` (e.g. 50), then press `S` to save.

### Step 2: Run Training
Place your raw text in `assets/data.txt` and run:

```bash
make train
./train assets/data.txt assets/vocab.txt model.bin
```

### Step 3: Run Your New Model
```bash
neuricode
```


---



## 📜 License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
