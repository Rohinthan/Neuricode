CC      = gcc
NVCC    ?= nvcc
CUDA_PATH ?= /usr/local/cuda

CFLAGS  = -O2 -Wall -Wextra -std=c11 -fPIC -fopenmp
CFLAGS += -I include/
CFLAGS += -I cuda/include/
CFLAGS += -I config/
CFLAGS += -I memory/
CFLAGS += -I .
LDFLAGS = -lm -fopenmp

# GPU_LDFLAGS is deliberately kept SEPARATE from LDFLAGS — it must
# only be added to the binaries that actually link GPU_OBJS (see
# below). If it were folded into the global LDFLAGS, `make config`
# (which links the standalone `menuconfig` tool) would fail on any
# machine without libcudart/libOpenCL installed — exactly the
# machine you need menuconfig on to turn GPU support OFF in the
# first place.
GPU_LDFLAGS =
GPU_OBJS    =

# ── auto-load neuralc_config.h if it exists ───────────────────────
ifneq (,$(wildcard neuralc_config.h))
  CFLAGS += -DNEURALC_HAS_CONFIG
  NEURALC_USE_OMP     := $(shell grep 'NEURALC_USE_OMP '     neuralc_config.h | awk '{print $$3}')
  NEURALC_USE_GPU     := $(shell grep 'NEURALC_USE_GPU '     neuralc_config.h | awk '{print $$3}')
  NEURALC_OPT         := $(shell grep 'NEURALC_OPT_LEVEL'    neuralc_config.h | awk '{print $$3}' | tr -d '"')
  ifeq ($(NEURALC_USE_OMP),1)
    CFLAGS  += -DUSE_OMP
  endif
  ifeq ($(NEURALC_USE_GPU),1)
    CFLAGS      += -DUSE_CUDA -I cuda/include -I cuda/ -I include/gpu
    GPU_LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lstdc++
    GPU_OBJS    += build/cuda_backend.o
  endif
  ifneq ($(NEURALC_OPT),)
    CFLAGS := $(filter-out -O2,$(CFLAGS)) $(NEURALC_OPT)
  endif
  $(info [neuralc] Config loaded - OMP=$(NEURALC_USE_OMP) GPU=$(NEURALC_USE_GPU) OPT=$(NEURALC_OPT))
endif

CONFIG_DEP = $(wildcard neuralc_config.h)

# ── sources ───────────────────────────────────────────────────────
SRC_CORE = src/tensor.c      \
           src/layer.c       \
           src/nn.c          \
           src/optimizer.c   \
           src/dataloader.c  \
           src/dropout.c     \
           src/batchnorm.c   \
           src/conv.c        \
           src/rnn.c         \
           src/transformer.c \
           src/sampler.c     \
           src/pipeline.c    \
           src/mnist.c	     \
           src/tokenizer.c   \
           src/tui.c         \
           src/dataset_loader.c


SRC_MEM  = memory/memory.c

SRC_CFG  = config/neuralc_init.c
SRC      = $(SRC_CORE) $(SRC_CFG) $(SRC_MEM)

OBJ_CORE = $(patsubst src/%.c,    build/%.o, $(SRC_CORE))
OBJ_CFG  = $(patsubst config/%.c, build/%.o, $(SRC_CFG))
OBJ_MEM  = $(patsubst memory/%.c, build/%.o, $(SRC_MEM))
# GPU_OBJS (cuda_backend.o / opencl_backend.o) is populated above,
# per whatever NEURALC_GPU_BACKEND says — empty when GPU is Off.
OBJ      = $(OBJ_CORE) $(OBJ_CFG) $(OBJ_MEM) $(GPU_OBJS)

.PHONY: all demo rnn_demo mnist_demo demo_mnist cnn_mnist demo_char_rnn \
        demo_char_rnn_asan config clean libneuralc omp test neuricode install

$(shell mkdir -p build)

# ── main targets ──────────────────────────────────────────────────
# NOTE: $(GPU_LDFLAGS) is appended explicitly here (and nowhere near
# the menuconfig rule below) — see the comment on GPU_LDFLAGS above.
all: neuralc demo rnn_demo mnist_demo neuricode

neuralc: $(OBJ) build/main.o
	$(CC) $(CFLAGS) -o neuralc $(OBJ) build/main.o $(LDFLAGS) $(GPU_LDFLAGS)

demo: $(OBJ) build/demo.o
	$(CC) $(CFLAGS) -o demo $(OBJ) build/demo.o $(LDFLAGS) $(GPU_LDFLAGS)

rnn_demo: $(OBJ) build/demo_rnn.o
	$(CC) $(CFLAGS) -o rnn_demo $(OBJ) build/demo_rnn.o $(LDFLAGS) $(GPU_LDFLAGS)

mnist_demo demo_mnist: $(OBJ) build/demo_mnist.o
	$(CC) $(CFLAGS) -o mnist_demo $(OBJ) build/demo_mnist.o $(LDFLAGS) $(GPU_LDFLAGS)

cnn_mnist: $(OBJ) build/demo_cnn_mnist.o
	$(CC) $(CFLAGS) -o cnn_mnist $(OBJ) build/demo_cnn_mnist.o $(LDFLAGS) $(GPU_LDFLAGS)

demo_char_rnn: $(OBJ) build/demo_char_rnn.o
	$(CC) $(CFLAGS) -o demo_char_rnn $(OBJ) build/demo_char_rnn.o $(LDFLAGS) $(GPU_LDFLAGS)
	
# NOTE: pipeline.c is a library module (Pipeline/pipeline_load/
# pipeline_forward/...) — it has no main() of its own, so it is
# compiled as part of $(OBJ) via SRC_CORE above (moved from apps/ to
# src/ — see the SRC_CORE list) and linked INTO cli/train/sanity
# below, exactly like tensor.c or layer.c. It is never its own
# executable target.

cli: apps/cli.c $(OBJ)
	$(CC) $(CFLAGS) -o cli apps/cli.c $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)

neuricode: apps/neuricode_cli.c $(OBJ)
	$(CC) $(CFLAGS) -o neuricode apps/neuricode_cli.c $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)

install: neuricode
	mkdir -p ~/.local/bin
	cp neuricode ~/.local/bin/neuricode
	@echo "Successfully installed neuricode to ~/.local/bin/neuricode"

train: apps/train.c $(OBJ)
	$(CC) $(CFLAGS) -o train apps/train.c $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)
	
# Diagnostic tool — kept as its own target/binary, separate from the
# production `cli` above, so it's never accidentally shipped or run
# in place of it. Links $(OBJ) exactly like cli/train (this was
# previously missing from the command line, even though $(OBJ) was
# already a prerequisite — every symbol from pipeline.c/tensor.c/
# tokenizer.c/etc. was undefined at link time until this was added).
sanity: tools/sanity_test.c $(OBJ)
	$(CC) $(CFLAGS) -o sanity tools/sanity_test.c $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)

test_transformer: tools/test_transformer.c $(OBJ)
	$(CC) $(CFLAGS) -o test_transformer tools/test_transformer.c $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)

benchmark_suite: tools/benchmark_suite.c $(OBJ)
	$(CC) $(CFLAGS) -o benchmark_suite tools/benchmark_suite.c $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)

# ── debug/ASan build ────────────────────────────────────────────────
# Separate from the normal demo_char_rnn target above: compiles every
# source file fresh with -fsanitize=address -g -O0 in one shot (no
# reused build/*.o — those were compiled without ASan instrumentation,
# and ASan requires every translation unit to be built consistently).
# Use this when debugging a crash: it prints the exact file/line of a
# memory error instead of a bare "Segmentation fault".
demo_char_rnn_asan: $(SRC) examples/demo_char_rnn.c
	$(CC) -g -O0 -fsanitize=address -Wall -Wextra -std=c11 -fPIC \
	      -I include/ -I config/ -I memory/ -I . -DNEURALC_HAS_CONFIG -DUSE_OMP -fopenmp \
	      $(SRC) examples/demo_char_rnn.c \
	      -o demo_char_rnn_asan -lm -fopenmp
	@echo "Built demo_char_rnn_asan — run ./demo_char_rnn_asan"

# ── explicit compile rules for all example files ──────────────────
build/main.o: examples/main.c
	$(CC) $(CFLAGS) -c examples/main.c -o build/main.o

build/demo.o: examples/demo.c
	$(CC) $(CFLAGS) -c examples/demo.c -o build/demo.o

build/demo_rnn.o: examples/demo_rnn.c
	$(CC) $(CFLAGS) -c examples/demo_rnn.c -o build/demo_rnn.o

build/demo_mnist.o: examples/demo_mnist.c
	$(CC) $(CFLAGS) -c examples/demo_mnist.c -o build/demo_mnist.o

build/demo_cnn_mnist.o: examples/demo_cnn_mnist.c
	$(CC) $(CFLAGS) -c examples/demo_cnn_mnist.c -o build/demo_cnn_mnist.o

build/demo_char_rnn.o: examples/demo_char_rnn.c
	$(CC) $(CFLAGS) -c examples/demo_char_rnn.c -o build/demo_char_rnn.o

# ── tests ─────────────────────────────────────────────────────────
test: $(OBJ) build/test_autograd.o
	$(CC) $(CFLAGS) -o test_tensor $(OBJ) build/test_autograd.o $(LDFLAGS) $(GPU_LDFLAGS)
	@echo "Running tests..."
	@./test_tensor

# ── menuconfig ────────────────────────────────────────────────────
# Deliberately does NOT use $(GPU_LDFLAGS) or link $(GPU_OBJS) — this
# tool must build on a machine with no CUDA/OpenCL toolchain at all,
# since picking "Off" is one of the things it's for. config_ui.c only
# probes for hardware via file-existence checks, it never calls into
# cuda_backend.h/opencl_backend.h.
config: menuconfig
	@./menuconfig

menuconfig: build/config_ui.o build/neuralc_config_main.o
	$(CC) $(CFLAGS) -o menuconfig \
	      build/config_ui.o build/neuralc_config_main.o -lm

build/config_ui.o: config/config_ui.c config/config_ui.h
	$(CC) $(CFLAGS) -c config/config_ui.c -o build/config_ui.o

build/neuralc_config_main.o: config/neuralc_config_main.c config/config_ui.h
	$(CC) $(CFLAGS) -c config/neuralc_config_main.c \
	      -o build/neuralc_config_main.o

# ── compile rules ─────────────────────────────────────────────────
build/%.o: src/%.c $(CONFIG_DEP)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: config/%.c $(CONFIG_DEP)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: memory/%.c $(CONFIG_DEP)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: examples/%.c $(CONFIG_DEP)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: tests/%.c $(CONFIG_DEP)
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: apps/%.c $(CONFIG_DEP)
	$(CC) $(CFLAGS) -c $< -o $@
	
# ── GPU backend objects ─────────────────────────────────────────────
# Only built when NEURALC_GPU_BACKEND selects them (see auto-config
# block above) — these explicit rules take precedence over the
# generic src/%.c / src/%.cu pattern since there is none for .cu.
build/cuda_backend.o: cuda/src/cuda_backend.cu cuda/include/cuda_backend.h
	$(NVCC) -O3 -Xcompiler -fPIC -I cuda/include -I include/ -c cuda/src/cuda_backend.cu -o build/cuda_backend.o

build/opencl_backend.o: src/gpu/opencl_backend.c include/gpu/opencl_backend.h
	$(CC) $(CFLAGS) -c src/gpu/opencl_backend.c -o build/opencl_backend.o

# ── shared library ────────────────────────────────────────────────
libneuralc: $(OBJ)
	$(CC) $(CFLAGS) -fPIC -shared -o libneuralc.so $(OBJ) $(LDFLAGS) $(GPU_LDFLAGS)
	@echo "Built libneuralc.so"

# ── omp ───────────────────────────────────────────────────────────
# Force OpenMP on without a config file. There's no equivalent `gpu:`
# convenience target any more — GPU backend selection now always
# goes through `make config` (NEURALC_GPU_BACKEND), since a plain
# `make gpu` can't express "CUDA vs OpenCL vs Both" as a single flag.
omp: CFLAGS  += -DUSE_OMP -fopenmp
omp: LDFLAGS += -fopenmp
omp: all

# ── clean ─────────────────────────────────────────────────────────
clean:
	rm -f build/*.o
	rm -f neuralc demo rnn_demo mnist_demo cnn_mnist test_tensor demo_char_rnn demo_char_rnn_asan 
	rm -f train cli sanity neurix
	rm -f neuricode
	rm -f benchmark_suite test_transformer
	rm -f menuconfig libneuralc.so test_transformer_bin
	rm -f xor_weights.bin mnist_best.bin cnn_mnist_best.bin
	@echo "Clean done"
