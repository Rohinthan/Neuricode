#ifndef NEURALC_GPU_H
#define NEURALC_GPU_H

/*
 * neuralc_gpu.h — OpenCL GPU backend for neuralc
 *
 * Requires:  OpenCL 1.2+
 * Install:   sudo apt install opencl-headers ocl-icd-opencl-dev
 * Build:     make gpu
 *
 * How it works:
 *   1. gpu_init()         — find GPU, compile kernels, create command queue
 *   2. gpu_tensor_upload  — copy CPU Tensor → GPU buffer
 *   3. gpu_*()            — run kernel on GPU
 *   4. gpu_tensor_download — copy GPU result → CPU Tensor
 *   5. gpu_shutdown()     — free all OpenCL resources
 *
 * Design: GPU buffers are separate from CPU Tensors.
 * You upload data, run GPU ops, then download results.
 * This avoids modifying the existing tensor.h API.
 */

#ifdef USE_OPENCL
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#include "../tensor.h"

/* ── GPU context (one per process) ─────────────────────────────────── */
typedef struct {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;

    /* compiled kernels */
    cl_kernel k_add;
    cl_kernel k_sub;
    cl_kernel k_mul;
    cl_kernel k_scale;
    cl_kernel k_relu;
    cl_kernel k_relu_grad;
    cl_kernel k_sigmoid;
    cl_kernel k_sigmoid_grad;
    cl_kernel k_tanh;
    cl_kernel k_matmul;
    cl_kernel k_matmul_tiled;
    cl_kernel k_dense_fwd;
    cl_kernel k_softmax;
} GPUContext;

/* ── GPU tensor (buffer on device) ─────────────────────────────────── */
typedef struct {
    cl_mem  buf;       /* OpenCL device buffer             */
    size_t  size;      /* number of float elements         */
    int     shape[8];
    int     ndim;
} GPUTensor;

/* ── lifecycle ──────────────────────────────────────────────────────── */

/*
 * gpu_init: initialize OpenCL, select GPU, compile kernels.
 * kernel_path: path to kernels.cl file (e.g. "gpu/kernels.cl")
 * Returns 0 on success, -1 on failure.
 */
int  gpu_init(GPUContext *ctx, const char *kernel_path);
void gpu_shutdown(GPUContext *ctx);
void gpu_print_device_info(const GPUContext *ctx);

/* ── memory ─────────────────────────────────────────────────────────── */

/* Allocate a GPU buffer of `size` floats */
GPUTensor *gpu_tensor_alloc(GPUContext *ctx, const int *shape, int ndim);

/* Upload CPU Tensor → GPU */
GPUTensor *gpu_tensor_upload(GPUContext *ctx, const Tensor *cpu_t);

/* Download GPU → CPU Tensor (caller allocates cpu_out) */
void gpu_tensor_download(GPUContext *ctx, const GPUTensor *gpu_t,
                         Tensor *cpu_out);

void gpu_tensor_free(GPUTensor *t);

/* ── GPU ops ────────────────────────────────────────────────────────── */

void gpu_add(GPUContext *ctx,
             const GPUTensor *a, const GPUTensor *b, GPUTensor *out);

void gpu_sub(GPUContext *ctx,
             const GPUTensor *a, const GPUTensor *b, GPUTensor *out);

void gpu_mul(GPUContext *ctx,
             const GPUTensor *a, const GPUTensor *b, GPUTensor *out);

void gpu_scale(GPUContext *ctx,
               const GPUTensor *a, float s, GPUTensor *out);

void gpu_relu(GPUContext *ctx,
              const GPUTensor *in, GPUTensor *out);

void gpu_relu_grad(GPUContext *ctx,
                   const GPUTensor *in, const GPUTensor *grad,
                   GPUTensor *out);

void gpu_sigmoid(GPUContext *ctx,
                 const GPUTensor *in, GPUTensor *out);

void gpu_sigmoid_grad(GPUContext *ctx,
                      const GPUTensor *sig, const GPUTensor *grad,
                      GPUTensor *out);

void gpu_tanh(GPUContext *ctx,
              const GPUTensor *in, GPUTensor *out);

/*
 * gpu_matmul: a[M,K] @ b[K,N] → out[M,N]
 * Uses tiled kernel for large matrices (faster).
 */
void gpu_matmul(GPUContext *ctx,
                const GPUTensor *a, const GPUTensor *b,
                GPUTensor *out, int M, int K, int N);

/*
 * gpu_dense_forward:
 *   input  [batch, in_f]
 *   W      [out_f, in_f]
 *   bias   [out_f]
 *   output [batch, out_f]
 */
void gpu_dense_forward(GPUContext *ctx,
                       const GPUTensor *input,
                       const GPUTensor *W,
                       const GPUTensor *bias,
                       GPUTensor *output,
                       int batch, int in_f, int out_f);

void gpu_softmax(GPUContext *ctx,
                 const GPUTensor *in, GPUTensor *out,
                 int rows, int cols);

/* ── sync ───────────────────────────────────────────────────────────── */
/* Wait for all GPU operations to finish */
void gpu_sync(GPUContext *ctx);

/* ── error helper ───────────────────────────────────────────────────── */
const char *gpu_cl_error_string(int err);

#else  /* USE_OPENCL not defined */

/* Stub so code compiles without OpenCL installed */
typedef struct { int dummy; } GPUContext;
typedef struct { int dummy; } GPUTensor;

static inline int  gpu_init(GPUContext *c, const char *p)
    { (void)c;(void)p; return -1; }
static inline void gpu_shutdown(GPUContext *c) { (void)c; }

#endif /* USE_OPENCL */
#endif /* NEURALC_GPU_H */
