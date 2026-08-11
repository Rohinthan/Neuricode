#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

/*
 * cuda_backend.h — C-callable interface to high-performance CUDA backend
 * Located in cuda/include/cuda_backend.h
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Stream & Multi-GPU Device Management ───────────────────────── */
int    cuda_available(void);                  /* 1 if CUDA device present */
int    cuda_get_device_count(void);          /* Number of available GPUs */
void   cuda_set_device(int device_id);       /* Switch active GPU device */
int    cuda_get_device(void);                /* Query active GPU device ID */

void  *cuda_stream_create(void);              /* Create new cudaStream_t */
void   cuda_stream_destroy(void *stream);     /* Destroy cudaStream_t */
void   cuda_set_stream(void *stream);         /* Set active stream for ops */
void  *cuda_get_stream(void);                 /* Get currently active stream */

void   cuda_enable_tensor_cores(int enable);  /* Toggle Tensor Core math */

/* ── Device & Pinned Host Memory Management ──────────────────────── */
float *cuda_malloc_f32(size_t n);             /* Device memory alloc (n floats) */
void   cuda_free_f32(float *dev_ptr);
float *cuda_malloc_host(size_t n);            /* Pinned host memory alloc */
void   cuda_free_host(float *host_ptr);

void   cuda_memcpy_h2d(float *dst_dev, const float *src_host, size_t n);
void   cuda_memcpy_d2h(float *dst_host, const float *src_dev, size_t n);
void   cuda_memcpy_d2d(float *dst_dev, const float *src_dev, size_t n);

void   cuda_memcpy_h2d_async(float *dst_dev, const float *src_host, size_t n, void *stream);
void   cuda_memcpy_d2h_async(float *dst_host, const float *src_dev, size_t n, void *stream);
void   cuda_memcpy_d2d_async(float *dst_dev, const float *src_dev, size_t n, void *stream);

void   cuda_memset_zero(float *dev_ptr, size_t n);
void   cuda_sync(void);                       /* Stream / Device sync */

/* ── Reusable GPU Memory Pool Workspace ──────────────────────────── */
typedef struct GpuMemoryPool GpuMemoryPool;

void   cuda_pool_init(size_t bytes);          /* Pre-allocate GPU workspace pool */
void  *cuda_pool_alloc(size_t bytes);         /* Fast offset allocation from pool */
void   cuda_pool_reset(void);                 /* Reset pool offset for next iteration */
void   cuda_pool_destroy(void);               /* Release pool memory */
void   cuda_pool_info(size_t *capacity, size_t *used, size_t *peak);

/* Memory Footprint & VRAM Diagnostic Helpers */
size_t estimate_dense_vram_bytes(int batch, int in_f, int out_f);
void   cuda_get_vram_info(size_t *free_bytes, size_t *total_bytes);

/* ── Linear Algebra & cuBLAS GEMM ────────────────────────────────── */
/* out[M,N] = a[M,K] @ b[K,N] (row-major, cuBLAS stream bound) */
void cuda_matmul(const float *a, const float *b, float *out,
                  int M, int K, int N);

/* Tiled shared-memory matrix transpose (32x32 tiles, bank conflict free) */
void cuda_transpose(const float *a, float *out, int rows, int cols);

/* ── Optimized Dense (Linear) Layer (cuBLAS GEMM Based) ──────────── */
/* Forward: z = X @ W^T + bias (X: BxI, W: OxI, z: BxO) */
void cuda_linear_forward(const float *x, const float *W, const float *bias,
                          float *z, int batch, int in_f, int out_f);

/* Backward: dW = (1/B)*dZ^T @ X, dX = dZ @ W, db = col_sum(dZ)/B */
void cuda_linear_backward(const float *x, const float *W, const float *dz,
                           float *dW, float *db, float *dX,
                           int batch, int in_f, int out_f);

/* ── Fused Kernels ──────────────────────────────────────────────── */
/* Single-pass linear forward + bias + ReLU activation */
void cuda_fused_linear_relu(const float *x, const float *W, const float *bias,
                             float *out, int batch, int in_f, int out_f);

/* Fused row-wise bias addition + activation (In-place or Out-of-place) */
void cuda_fused_bias_add(float *z, const float *bias, int batch, int cols);
void cuda_fused_bias_relu(float *z, const float *bias, int batch, int cols);
void cuda_fused_bias_gelu(float *z, const float *bias, int batch, int cols);
void cuda_fused_bias_silu(float *z, const float *bias, int batch, int cols);

/* Elementwise fused ops */
void cuda_fused_add_relu(const float *a, const float *b, float *out, size_t n);
void cuda_fused_mul_relu(const float *a, const float *b, float *out, size_t n);
void cuda_fused_add_bias_relu(const float *a, const float *bias, float *out,
                              int batch, int cols);

/* ── Elementwise Ops ─────────────────────────────────────────────── */
void cuda_add(const float *a, const float *b, float *out, size_t n);
void cuda_sub(const float *a, const float *b, float *out, size_t n);
void cuda_mul(const float *a, const float *b, float *out, size_t n);
void cuda_scale(const float *a, float s, float *out, size_t n);
void cuda_add_scalar(const float *a, float s, float *out, size_t n);
void cuda_fill(float *a, float val, size_t n);

/* ── Activations & Warp-Level Softmax ────────────────────────────── */
void cuda_relu(const float *a, float *out, size_t n);
void cuda_relu_grad(const float *a, const float *grad, float *out, size_t n);
void cuda_sigmoid(const float *a, float *out, size_t n);
void cuda_sigmoid_grad(const float *sig, const float *grad, float *out, size_t n);
void cuda_tanh_f(const float *a, float *out, size_t n);
void cuda_tanh_grad(const float *th, const float *grad, float *out, size_t n);

/* High-efficiency warp-shfl optimized Softmax */
void cuda_softmax_rows(const float *a, float *out, int rows, int cols);

/* ── Zero-Padding & Conv2D ───────────────────────────────────────── */
void cuda_pad2d(const float *input, float *output,
                 int batch, int C, int H, int W, int pad);
void cuda_unpad2d(const float *padded, float *output,
                   int batch, int C, int H, int W, int pad);

void cuda_conv2d_forward(const float *input_padded, const float *W,
                          const float *bias, float *output,
                          int batch, int in_C, int pH, int pW,
                          int out_C, int kH, int kW, int stride,
                          int out_H, int out_W);

void cuda_conv2d_backward(const float *input_padded, const float *W,
                           const float *grad_out,
                           float *dW, float *db, float *dpad_grad_in,
                           int batch, int in_C, int pH, int pW,
                           int out_C, int kH, int kW, int stride,
                           int out_H, int out_W);

/* ── MaxPool2D ───────────────────────────────────────────────────── */
void cuda_maxpool2d_forward(const float *input, float *output, float *mask,
                             int batch, int C, int H, int W,
                             int pool_size, int stride, int out_H, int out_W);

void cuda_maxpool2d_backward(const float *grad_out, float *grad_in,
                              const float *mask,
                              int batch, int C, int H, int W,
                              int pool_size, int stride, int out_H, int out_W);

/* ── FP16 Half-Precision & Tensor Core Accelerated Operations ─────── */
void *cuda_malloc_f16(size_t n);
void  cuda_free_f16(void *dev_ptr);
void  cuda_f32_to_f16(const float *src_f32, void *dst_f16, size_t n);
void  cuda_f16_to_f32(const void *src_f16, float *dst_f32, size_t n);

void  cuda_matmul_fp16(const void *a_f16, const void *b_f16, void *out_f16,
                       int M, int K, int N);

void  cuda_linear_forward_fp16(const void *x_f16, const void *w_f16, const float *bias,
                              void *z_f16, int batch, int in_f, int out_f);

void  cuda_linear_backward_fp16(const void *x_f16, const void *w_f16, const void *dz_f16,
                               void *dw_f16, float *db_f32, void *dx_f16,
                               int batch, int in_f, int out_f);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_BACKEND_H */
