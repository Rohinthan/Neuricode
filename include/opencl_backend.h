#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

/*
 * opencl_backend.h
 * ────────────────────────────────────────────────────────────────
 * OpenCL counterpart to cuda_backend.h, for low-end / non-NVIDIA
 * hardware (Intel and AMD integrated graphics, older GPUs, anything
 * with an OpenCL ICD installed). Compiled with plain gcc and linked
 * with -lOpenCL — no special compiler required, unlike CUDA's nvcc.
 *
 * SCOPE (intentionally narrower than the CUDA backend for now):
 *   matmul, dense/linear forward+backward, elementwise ops, and
 *   activations. No Conv2D / MaxPool2D yet — those stay CPU-only
 *   when CF_GPU_OPENCL is selected. This is enough to run a full
 *   dense-only network (e.g. an MLP on MNIST) on the GPU to verify
 *   the OpenCL pipeline actually works end to end; conv/pool kernels
 *   can be added the same way once this path is confirmed working.
 *
 * All functions operate on `float*` parameters that are really a
 * `cl_mem` handle reinterpret-cast to float* (cl_mem is itself just
 * an opaque pointer type, so the cast is value-preserving) — this
 * keeps tensor.h/tensor.c free of any OpenCL-specific types, exactly
 * like the CUDA backend keeps them free of CUDA types.
 *
 * Build: gcc -c opencl_backend.c -o opencl_backend.o
 * Link:  -lOpenCL
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── device / memory management ─────────────────────────────────── */
int    opencl_available(void);          /* 1 if an OpenCL device is present */
float *opencl_malloc_f32(size_t n);     /* device alloc, n floats           */
void   opencl_free_f32(float *dev_buf);
void   opencl_memcpy_h2d(float *dst_dev, const float *src_host, size_t n);
void   opencl_memcpy_d2h(float *dst_host, const float *src_dev, size_t n);
void   opencl_memcpy_d2d(float *dst_dev, const float *src_dev, size_t n);
void   opencl_memset_zero(float *dev_buf, size_t n);
void   opencl_sync(void);

/* Human-readable name of the selected OpenCL device, e.g. for the
 * config_ui hardware banner. Returns a pointer to an internal static
 * buffer — copy it if you need it to outlive the next call.        */
const char *opencl_device_name(void);

/* ── linear algebra ─────────────────────────────────────────────── */
/* out[M,N] = a[M,K] @ b[K,N]   (row-major, all device pointers)     */
void opencl_matmul(const float *a, const float *b, float *out,
                    int M, int K, int N);

/* ── dense (fully-connected) layer ──────────────────────────────── */
/* z[b,o] = sum_i x[b,i] * W[o,i] + bias[o]      (W is [out,in])     */
void opencl_linear_forward(const float *x, const float *W, const float *bias,
                            float *z, int batch, int in_f, int out_f);

/* Same semantics as cuda_linear_backward(): dW/db are the batch mean,
 * dX is the raw sum over out_f. dW/db/dX must be pre-allocated.     */
void opencl_linear_backward(const float *x, const float *W, const float *dz,
                             float *dW, float *db, float *dX,
                             int batch, int in_f, int out_f);

/* ── elementwise ops ─────────────────────────────────────────────── */
void opencl_add(const float *a, const float *b, float *out, size_t n);
void opencl_sub(const float *a, const float *b, float *out, size_t n);
void opencl_mul(const float *a, const float *b, float *out, size_t n);
void opencl_scale(const float *a, float s, float *out, size_t n);
void opencl_add_scalar(const float *a, float s, float *out, size_t n);
void opencl_fill(float *a, float val, size_t n);

/* ── activations ─────────────────────────────────────────────────── */
void opencl_relu(const float *a, float *out, size_t n);
void opencl_relu_grad(const float *a, const float *grad, float *out, size_t n);
void opencl_sigmoid(const float *a, float *out, size_t n);
void opencl_sigmoid_grad(const float *sig, const float *grad, float *out, size_t n);
void opencl_tanh_f(const float *a, float *out, size_t n);
void opencl_tanh_grad(const float *th, const float *grad, float *out, size_t n);
void opencl_softmax_rows(const float *a, float *out, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* OPENCL_BACKEND_H */
