#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

/*
 * cuda_backend.h
 * ────────────────────────────────────────────────────────────────
 * C-callable interface to the CUDA kernels in cuda_backend.cu.
 *
 * This header contains NO CUDA syntax, so it is safe to #include
 * from plain .c files compiled with gcc — only cuda_backend.cu
 * itself is compiled with nvcc. Every function here operates on
 * device pointers (float* living in GPU memory) unless documented
 * otherwise.
 *
 * All functions call exit(1) via cuda_check() on any CUDA error,
 * matching the library's existing cforge_error() fail-fast style.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── device / memory management ─────────────────────────────────── */
int    cuda_available(void);           /* 1 if a CUDA device is present */
float *cuda_malloc_f32(size_t n);      /* device alloc, n floats        */
void   cuda_free_f32(float *dev_ptr);
void   cuda_memcpy_h2d(float *dst_dev, const float *src_host, size_t n);
void   cuda_memcpy_d2h(float *dst_host, const float *src_dev, size_t n);
void   cuda_memcpy_d2d(float *dst_dev, const float *src_dev, size_t n);
void   cuda_memset_zero(float *dev_ptr, size_t n);
void   cuda_sync(void);

/* ── linear algebra ─────────────────────────────────────────────── */
/* out[M,N] = a[M,K] @ b[K,N]   (row-major, all device pointers)     */
void cuda_matmul(const float *a, const float *b, float *out,
                  int M, int K, int N);
/* out[cols,rows] = transpose(a[rows,cols]) */
void cuda_transpose(const float *a, float *out, int rows, int cols);

/* ── dense (fully-connected) layer ──────────────────────────────── */
/* z[b,o] = sum_i x[b,i] * W[o,i] + bias[o]      (W is [out,in])     */
void cuda_linear_forward(const float *x, const float *W, const float *bias,
                          float *z, int batch, int in_f, int out_f);

/* Given dz = dL/dz [batch,out_f]:
 *   dW[o,i] = (1/batch) * sum_b dz[b,o]*x[b,i]
 *   db[o]   = (1/batch) * sum_b dz[b,o]
 *   dX[b,i] = sum_o dz[b,o]*W[o,i]
 * dW/db/dX must be pre-allocated by the caller (dW/db need not be
 * pre-zeroed — this call overwrites them).
 */
void cuda_linear_backward(const float *x, const float *W, const float *dz,
                           float *dW, float *db, float *dX,
                           int batch, int in_f, int out_f);

/* ── elementwise ops ─────────────────────────────────────────────── */
void cuda_add(const float *a, const float *b, float *out, size_t n);
void cuda_sub(const float *a, const float *b, float *out, size_t n);
void cuda_mul(const float *a, const float *b, float *out, size_t n);
void cuda_scale(const float *a, float s, float *out, size_t n);
void cuda_add_scalar(const float *a, float s, float *out, size_t n);
void cuda_fill(float *a, float val, size_t n);

/* ── activations ─────────────────────────────────────────────────── */
void cuda_relu(const float *a, float *out, size_t n);
void cuda_relu_grad(const float *a, const float *grad, float *out, size_t n);
void cuda_sigmoid(const float *a, float *out, size_t n);
void cuda_sigmoid_grad(const float *sig, const float *grad, float *out, size_t n);
void cuda_tanh_f(const float *a, float *out, size_t n);
void cuda_tanh_grad(const float *th, const float *grad, float *out, size_t n);
void cuda_softmax_rows(const float *a, float *out, int rows, int cols);

/* ── zero-padding helpers for Conv2D ────────────────────────────────
 * cuda_pad2d:   input [batch,C,H,W] -> output [batch,C,H+2p,W+2p],
 *               output must be pre-zeroed by the caller (or use
 *               cuda_memset_zero first) since only the interior is
 *               written.
 * cuda_unpad2d: inverse crop, padded [batch,C,H+2p,W+2p] -> out
 *               [batch,C,H,W] (plain copy, no accumulation).       */
void cuda_pad2d(const float *input, float *output,
                 int batch, int C, int H, int W, int pad);
void cuda_unpad2d(const float *padded, float *output,
                   int batch, int C, int H, int W, int pad);

/* ── Conv2D (direct convolution, input already zero-padded) ────────
 * input_padded [batch, in_C, pH, pW]
 * W            [out_C, in_C, kH, kW]
 * bias         [out_C]
 * output       [batch, out_C, out_H, out_W]
 */
void cuda_conv2d_forward(const float *input_padded, const float *W,
                          const float *bias, float *output,
                          int batch, int in_C, int pH, int pW,
                          int out_C, int kH, int kW, int stride,
                          int out_H, int out_W);

/* grad_out       [batch, out_C, out_H, out_W]   upstream gradient
 * dW, db         pre-allocated, overwritten with the *unscaled* sum
 *                over the batch (caller scales by 1/batch, matching
 *                the existing CPU conv2d_backward)
 * dpad_grad_in   [batch, in_C, pH, pW] pre-allocated, zeroed by this
 *                call before accumulation
 */
void cuda_conv2d_backward(const float *input_padded, const float *W,
                           const float *grad_out,
                           float *dW, float *db, float *dpad_grad_in,
                           int batch, int in_C, int pH, int pW,
                           int out_C, int kH, int kW, int stride,
                           int out_H, int out_W);

/* ── MaxPool2D ───────────────────────────────────────────────────── */
/* mask[out_idx] stores the flat index (as a float) of the argmax
 * element within the full `input` tensor — matches the existing CPU
 * maxpool2d_forward/backward exactly (mask is NOT a within-window
 * offset, it's a global input index, so backward is a pure scatter).
 */
void cuda_maxpool2d_forward(const float *input, float *output, float *mask,
                             int batch, int C, int H, int W,
                             int pool_size, int stride, int out_H, int out_W);

void cuda_maxpool2d_backward(const float *grad_out, float *grad_in,
                              const float *mask,
                              int batch, int C, int H, int W,
                              int pool_size, int stride, int out_H, int out_W);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_BACKEND_H */
