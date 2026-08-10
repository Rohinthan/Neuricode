/*
 * cuda/src/cuda_backend.cu
 * ────────────────────────────────────────────────────────────────
 * CUDA kernels + launchers backing cuda_backend.h in cuda/ directory.
 *
 * Compiled with nvcc into build/cuda_backend.o.
 */

#include "../include/cuda_backend.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/* ── error handling ─────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                        \
        if (_err != cudaSuccess) {                                        \
            fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                             \
            exit(1);                                                      \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _st = (call);                                       \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "[cuBLAS] %s:%d: status %d\n", __FILE__,       \
                    __LINE__, (int)_st);                                   \
            exit(1);                                                      \
        }                                                                   \
    } while (0)

#define THREADS 256
static inline int blocks_for(size_t n) {
    return (int)((n + THREADS - 1) / THREADS);
}

/* ── Stream & Device Global State ──────────────────────────────── */
static cudaStream_t g_active_stream = 0;
static cublasHandle_t g_cublas = NULL;
static int g_tensor_cores_enabled = 1;

static cublasHandle_t cublas_handle(void) {
    if (!g_cublas) {
        CUBLAS_CHECK(cublasCreate(&g_cublas));
#if CUBLAS_VER_MAJOR >= 9
        if (g_tensor_cores_enabled) {
            cublasSetMathMode(g_cublas, CUBLAS_TENSOR_OP_MATH);
        }
#endif
    }
    CUBLAS_CHECK(cublasSetStream(g_cublas, g_active_stream));
    return g_cublas;
}

/* ══════════════════════════════════════════════════════════════════
 * Device / Stream / Multi-GPU Control
 * ════════════════════════════════════════════════════════════════ */

int cuda_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

int cuda_get_device_count(void) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void cuda_set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

int cuda_get_device(void) {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
}

void *cuda_stream_create(void) {
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return (void *)stream;
}

void cuda_stream_destroy(void *stream) {
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy((cudaStream_t)stream));
    }
}

void cuda_set_stream(void *stream) {
    g_active_stream = (cudaStream_t)stream;
}

void *cuda_get_stream(void) {
    return (void *)g_active_stream;
}

void cuda_enable_tensor_cores(int enable) {
    g_tensor_cores_enabled = enable;
    if (g_cublas) {
#if CUBLAS_VER_MAJOR >= 9
        cublasSetMathMode(g_cublas, enable ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
#endif
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Memory Allocation & Transfers (Pinned Host & Async Device)
 * ════════════════════════════════════════════════════════════════ */

float *cuda_malloc_f32(size_t n) {
    float *ptr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&ptr, n * sizeof(float)));
    return ptr;
}

void cuda_free_f32(float *dev_ptr) {
    if (dev_ptr) CUDA_CHECK(cudaFree(dev_ptr));
}

float *cuda_malloc_host(size_t n) {
    float *ptr = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&ptr, n * sizeof(float)));
    return ptr;
}

void cuda_free_host(float *host_ptr) {
    if (host_ptr) CUDA_CHECK(cudaFreeHost(host_ptr));
}

void cuda_memcpy_h2d_async(float *dst_dev, const float *src_host, size_t n, void *stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst_dev, src_host, n * sizeof(float),
                               cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

void cuda_memcpy_d2h_async(float *dst_host, const float *src_dev, size_t n, void *stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst_host, src_dev, n * sizeof(float),
                               cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}

void cuda_memcpy_d2d_async(float *dst_dev, const float *src_dev, size_t n, void *stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst_dev, src_dev, n * sizeof(float),
                               cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
}

void cuda_memcpy_h2d(float *dst_dev, const float *src_host, size_t n) {
    cuda_memcpy_h2d_async(dst_dev, src_host, n, (void *)g_active_stream);
}

void cuda_memcpy_d2h(float *dst_host, const float *src_dev, size_t n) {
    cuda_memcpy_d2h_async(dst_host, src_dev, n, (void *)g_active_stream);
}

void cuda_memcpy_d2d(float *dst_dev, const float *src_dev, size_t n) {
    cuda_memcpy_d2d_async(dst_dev, src_dev, n, (void *)g_active_stream);
}

void cuda_memset_zero(float *dev_ptr, size_t n) {
    CUDA_CHECK(cudaMemsetAsync(dev_ptr, 0, n * sizeof(float), g_active_stream));
}

void cuda_sync(void) {
    if (g_active_stream) {
        CUDA_CHECK(cudaStreamSynchronize(g_active_stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Linear Algebra & Optimized Transpose
 * ════════════════════════════════════════════════════════════════ */

void cuda_matmul(const float *a, const float *b, float *out,
                  int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(),
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              b, N,
                              a, K,
                              &beta,
                              out, N));
}

/* Shared Memory Tiled Transpose (32x32 Tile with Padded Dimension for Zero Bank Conflicts) */
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void k_transpose_tiled(const float * __restrict__ in,
                                  float * __restrict__ out,
                                  int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < cols) {
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void cuda_transpose(const float *a, float *out, int rows, int cols) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    k_transpose_tiled<<<grid, block, 0, g_active_stream>>>(a, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

/* ══════════════════════════════════════════════════════════════════
 * Dense (Fully-Connected) Layer & Fused Forward Pass
 * ════════════════════════════════════════════════════════════════ */

__global__ void k_linear_forward(const float *x, const float *W,
                                  const float *bias, float *z,
                                  int batch, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_f) return;
    int b = idx / out_f;
    int o = idx % out_f;

    float acc = bias[o];
    const float *xr = x + (size_t)b * in_f;
    const float *wr = W + (size_t)o * in_f;
    for (int i = 0; i < in_f; i++)
        acc += xr[i] * wr[i];
    z[idx] = acc;
}

void cuda_linear_forward(const float *x, const float *W, const float *bias,
                          float *z, int batch, int in_f, int out_f) {
    size_t n = (size_t)batch * out_f;
    k_linear_forward<<<blocks_for(n), THREADS, 0, g_active_stream>>>(
        x, W, bias, z, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_fused_linear_relu(const float *x, const float *W,
                                     const float *bias, float *out,
                                     int batch, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_f) return;
    int b = idx / out_f;
    int o = idx % out_f;

    float acc = bias[o];
    const float *xr = x + (size_t)b * in_f;
    const float *wr = W + (size_t)o * in_f;

    int i = 0;
    for (; i <= in_f - 4; i += 4) {
        acc += xr[i]     * wr[i];
        acc += xr[i + 1] * wr[i + 1];
        acc += xr[i + 2] * wr[i + 2];
        acc += xr[i + 3] * wr[i + 3];
    }
    for (; i < in_f; i++) {
        acc += xr[i] * wr[i];
    }

    out[idx] = acc > 0.0f ? acc : 0.0f;
}

void cuda_fused_linear_relu(const float *x, const float *W, const float *bias,
                             float *out, int batch, int in_f, int out_f) {
    size_t n = (size_t)batch * out_f;
    k_fused_linear_relu<<<blocks_for(n), THREADS, 0, g_active_stream>>>(
        x, W, bias, out, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_linear_dW(const float *x, const float *dz, float *dW,
                             int batch, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f * in_f) return;
    int o = idx / in_f;
    int i = idx % in_f;

    float acc = 0.0f;
    for (int b = 0; b < batch; b++)
        acc += dz[(size_t)b * out_f + o] * x[(size_t)b * in_f + i];
    dW[idx] = acc / (float)batch;
}

__global__ void k_linear_db(const float *dz, float *db, int batch, int out_f) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_f) return;
    float acc = 0.0f;
    for (int b = 0; b < batch; b++)
        acc += dz[(size_t)b * out_f + o];
    db[o] = acc / (float)batch;
}

__global__ void k_linear_dX(const float *W, const float *dz, float *dX,
                             int batch, int in_f, int out_f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * in_f) return;
    int b = idx / in_f;
    int i = idx % in_f;

    float acc = 0.0f;
    for (int o = 0; o < out_f; o++)
        acc += dz[(size_t)b * out_f + o] * W[(size_t)o * in_f + i];
    dX[idx] = acc;
}

void cuda_linear_backward(const float *x, const float *W, const float *dz,
                           float *dW, float *db, float *dX,
                           int batch, int in_f, int out_f) {
    k_linear_dW<<<blocks_for((size_t)out_f * in_f), THREADS, 0, g_active_stream>>>(
        x, dz, dW, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());

    k_linear_db<<<blocks_for(out_f), THREADS, 0, g_active_stream>>>(
        dz, db, batch, out_f);
    CUDA_CHECK(cudaGetLastError());

    k_linear_dX<<<blocks_for((size_t)batch * in_f), THREADS, 0, g_active_stream>>>(
        W, dz, dX, batch, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());
}

/* ══════════════════════════════════════════════════════════════════
 * Elementwise Operations & Fused Kernels
 * ════════════════════════════════════════════════════════════════ */

#define ELEMWISE_BINARY(NAME, OP)                                          \
    __global__ void k_##NAME(const float *a, const float *b, float *out,   \
                              size_t n) {                                  \
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;          \
        if (i < n) out[i] = (OP);                                          \
    }                                                                      \
    void cuda_##NAME(const float *a, const float *b, float *out, size_t n) { \
        k_##NAME<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, b, out, n); \
        CUDA_CHECK(cudaGetLastError());                                    \
    }

ELEMWISE_BINARY(add, a[i] + b[i])
ELEMWISE_BINARY(sub, a[i] - b[i])
ELEMWISE_BINARY(mul, a[i] * b[i])

__global__ void k_fused_add_relu(const float *a, const float *b, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = a[i] + b[i];
        out[i] = sum > 0.0f ? sum : 0.0f;
    }
}

void cuda_fused_add_relu(const float *a, const float *b, float *out, size_t n) {
    k_fused_add_relu<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_fused_mul_relu(const float *a, const float *b, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float prod = a[i] * b[i];
        out[i] = prod > 0.0f ? prod : 0.0f;
    }
}

void cuda_fused_mul_relu(const float *a, const float *b, float *out, size_t n) {
    k_fused_mul_relu<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_fused_add_bias_relu(const float *a, const float *bias, float *out,
                                      int batch, int cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cols;
    if (idx >= total) return;
    int c = idx % cols;
    float val = a[idx] + bias[c];
    out[idx] = val > 0.0f ? val : 0.0f;
}

void cuda_fused_add_bias_relu(const float *a, const float *bias, float *out,
                              int batch, int cols) {
    size_t total = (size_t)batch * cols;
    k_fused_add_bias_relu<<<blocks_for(total), THREADS, 0, g_active_stream>>>(
        a, bias, out, batch, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_scale(const float *a, float s, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}
void cuda_scale(const float *a, float s, float *out, size_t n) {
    k_scale<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, s, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_add_scalar(const float *a, float s, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s;
}
void cuda_add_scalar(const float *a, float s, float *out, size_t n) {
    k_add_scalar<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, s, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_fill(float *a, float val, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = val;
}
void cuda_fill(float *a, float val, size_t n) {
    k_fill<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, val, n);
    CUDA_CHECK(cudaGetLastError());
}

/* ══════════════════════════════════════════════════════════════════
 * Activations
 * ════════════════════════════════════════════════════════════════ */

__global__ void k_relu(const float *a, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
void cuda_relu(const float *a, float *out, size_t n) {
    k_relu<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_relu_grad(const float *a, const float *grad, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] > 0.0f ? grad[i] : 0.0f;
}
void cuda_relu_grad(const float *a, const float *grad, float *out, size_t n) {
    k_relu_grad<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, grad, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sigmoid(const float *a, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}
void cuda_sigmoid(const float *a, float *out, size_t n) {
    k_sigmoid<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sigmoid_grad(const float *sig, const float *grad, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = grad[i] * sig[i] * (1.0f - sig[i]);
}
void cuda_sigmoid_grad(const float *sig, const float *grad, float *out, size_t n) {
    k_sigmoid_grad<<<blocks_for(n), THREADS, 0, g_active_stream>>>(sig, grad, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_tanh(const float *a, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(a[i]);
}
void cuda_tanh_f(const float *a, float *out, size_t n) {
    k_tanh<<<blocks_for(n), THREADS, 0, g_active_stream>>>(a, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_tanh_grad(const float *th, const float *grad, float *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = grad[i] * (1.0f - th[i] * th[i]);
}
void cuda_tanh_grad(const float *th, const float *grad, float *out, size_t n) {
    k_tanh_grad<<<blocks_for(n), THREADS, 0, g_active_stream>>>(th, grad, out, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_softmax_rows(const float *a, float *out, int cols) {
    extern __shared__ float sh[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *in_row = a + (size_t)row * cols;
    float *out_row = out + (size_t)row * cols;

    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, in_row[j]);
    sh[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    float row_max = sh[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(in_row[j] - row_max);
        out_row[j] = e;
        local_sum += e;
    }
    sh[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float row_sum = sh[0];
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x)
        out_row[j] /= row_sum;
}

void cuda_softmax_rows(const float *a, float *out, int rows, int cols) {
    int threads = 128;
    size_t shmem = threads * sizeof(float);
    k_softmax_rows<<<rows, threads, shmem, g_active_stream>>>(a, out, cols);
    CUDA_CHECK(cudaGetLastError());
}

/* ══════════════════════════════════════════════════════════════════
 * Conv2D — Direct Convolution & Padding
 * ════════════════════════════════════════════════════════════════ */

#define IDX4D(b,c,h,w, C,H,W) \
    ((size_t)(b)*(C)*(H)*(W) + (size_t)(c)*(H)*(W) + (size_t)(h)*(W) + (w))

__global__ void k_pad2d(const float *input, float *output,
                         int batch, int C, int H, int W, int pad) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / ((size_t)W * H)) % C;
    int b = idx / ((size_t)W * H * C);

    int pH = H + 2 * pad, pW = W + 2 * pad;
    output[IDX4D(b, c, h + pad, w + pad, C, pH, pW)] = input[idx];
}

void cuda_pad2d(const float *input, float *output,
                 int batch, int C, int H, int W, int pad) {
    size_t total = (size_t)batch * C * H * W;
    size_t pad_n = (size_t)batch * C * (H + 2 * pad) * (W + 2 * pad);
    cuda_memset_zero(output, pad_n);
    if (pad == 0) { cuda_memcpy_d2d(output, input, total); return; }
    k_pad2d<<<blocks_for(total), THREADS, 0, g_active_stream>>>(
        input, output, batch, C, H, W, pad);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_unpad2d(const float *padded, float *output,
                           int batch, int C, int H, int W, int pad) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / ((size_t)W * H)) % C;
    int b = idx / ((size_t)W * H * C);

    int pH = H + 2 * pad, pW = W + 2 * pad;
    output[idx] = padded[IDX4D(b, c, h + pad, w + pad, C, pH, pW)];
}

void cuda_unpad2d(const float *padded, float *output,
                   int batch, int C, int H, int W, int pad) {
    size_t total = (size_t)batch * C * H * W;
    if (pad == 0) { cuda_memcpy_d2d(output, padded, total); return; }
    k_unpad2d<<<blocks_for(total), THREADS, 0, g_active_stream>>>(
        padded, output, batch, C, H, W, pad);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_conv2d_forward(const float *input, const float *W,
                                  const float *bias, float *output,
                                  int batch, int in_C, int pH, int pW,
                                  int out_C, int kH, int kW, int stride,
                                  int out_H, int out_W) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * out_C * out_H * out_W;
    if (idx >= total) return;

    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / ((size_t)out_W * out_H)) % out_C;
    int b  = idx / ((size_t)out_W * out_H * out_C);

    float acc = bias[oc];
    for (int ic = 0; ic < in_C; ic++)
    for (int kh = 0; kh < kH; kh++)
    for (int kw = 0; kw < kW; kw++) {
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        acc += W[IDX4D(oc, ic, kh, kw, in_C, kH, kW)]
             * input[IDX4D(b, ic, ih, iw, in_C, pH, pW)];
    }
    output[idx] = acc;
}

void cuda_conv2d_forward(const float *input_padded, const float *W,
                          const float *bias, float *output,
                          int batch, int in_C, int pH, int pW,
                          int out_C, int kH, int kW, int stride,
                          int out_H, int out_W) {
    size_t total = (size_t)batch * out_C * out_H * out_W;
    k_conv2d_forward<<<blocks_for(total), THREADS, 0, g_active_stream>>>(
        input_padded, W, bias, output,
        batch, in_C, pH, pW, out_C, kH, kW, stride, out_H, out_W);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_conv2d_backward(const float *input, const float *W,
                                   const float *grad_out,
                                   float *dW, float *db, float *dpad,
                                   int batch, int in_C, int pH, int pW,
                                   int out_C, int kH, int kW, int stride,
                                   int out_H, int out_W) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * out_C * out_H * out_W;
    if (idx >= total) return;

    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int oc = (idx / ((size_t)out_W * out_H)) % out_C;
    int b  = idx / ((size_t)out_W * out_H * out_C);

    float g = grad_out[idx];
    atomicAdd(&db[oc], g);

    for (int ic = 0; ic < in_C; ic++)
    for (int kh = 0; kh < kH; kh++)
    for (int kw = 0; kw < kW; kw++) {
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        size_t widx = IDX4D(oc, ic, kh, kw, in_C, kH, kW);
        size_t iidx = IDX4D(b, ic, ih, iw, in_C, pH, pW);
        atomicAdd(&dW[widx], g * input[iidx]);
        atomicAdd(&dpad[iidx], g * W[widx]);
    }
}

void cuda_conv2d_backward(const float *input_padded, const float *W,
                           const float *grad_out,
                           float *dW, float *db, float *dpad_grad_in,
                           int batch, int in_C, int pH, int pW,
                           int out_C, int kH, int kW, int stride,
                           int out_H, int out_W) {
    size_t w_n = (size_t)out_C * in_C * kH * kW;
    size_t pad_n = (size_t)batch * in_C * pH * pW;

    cuda_memset_zero(dW, w_n);
    cuda_memset_zero(db, out_C);
    cuda_memset_zero(dpad_grad_in, pad_n);

    size_t total = (size_t)batch * out_C * out_H * out_W;
    k_conv2d_backward<<<blocks_for(total), THREADS, 0, g_active_stream>>>(
        input_padded, W, grad_out, dW, db, dpad_grad_in,
        batch, in_C, pH, pW, out_C, kH, kW, stride, out_H, out_W);
    CUDA_CHECK(cudaGetLastError());
}

/* ══════════════════════════════════════════════════════════════════
 * MaxPool2D
 * ════════════════════════════════════════════════════════════════ */

__global__ void k_maxpool2d_forward(const float *input, float *output,
                                     float *mask,
                                     int batch, int C, int H, int W,
                                     int pool_size, int stride,
                                     int out_H, int out_W) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * C * out_H * out_W;
    if (idx >= total) return;

    int ow = idx % out_W;
    int oh = (idx / out_W) % out_H;
    int c  = (idx / ((size_t)out_W * out_H)) % C;
    int b  = idx / ((size_t)out_W * out_H * C);

    float max_val = -FLT_MAX;
    size_t max_idx = 0;
    for (int kh = 0; kh < pool_size; kh++)
    for (int kw = 0; kw < pool_size; kw++) {
        int ih = oh * stride + kh;
        int iw = ow * stride + kw;
        size_t iidx = IDX4D(b, c, ih, iw, C, H, W);
        float v = input[iidx];
        if (v > max_val) { max_val = v; max_idx = iidx; }
    }
    output[idx] = max_val;
    mask[idx]   = (float)max_idx;
}

void cuda_maxpool2d_forward(const float *input, float *output, float *mask,
                             int batch, int C, int H, int W,
                             int pool_size, int stride, int out_H, int out_W) {
    size_t total = (size_t)batch * C * out_H * out_W;
    k_maxpool2d_forward<<<blocks_for(total), THREADS, 0, g_active_stream>>>(
        input, output, mask, batch, C, H, W, pool_size, stride, out_H, out_W);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_maxpool2d_backward(const float *grad_out, float *grad_in,
                                      const float *mask, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    size_t src = (size_t)mask[idx];
    atomicAdd(&grad_in[src], grad_out[idx]);
}

void cuda_maxpool2d_backward(const float *grad_out, float *grad_in,
                              const float *mask,
                              int batch, int C, int H, int W,
                              int pool_size, int stride, int out_H, int out_W) {
    (void)pool_size; (void)stride;
    size_t in_n = (size_t)batch * C * H * W;
    size_t out_n = (size_t)batch * C * out_H * out_W;

    cuda_memset_zero(grad_in, in_n);
    k_maxpool2d_backward<<<blocks_for(out_n), THREADS, 0, g_active_stream>>>(
        grad_out, grad_in, mask, out_n);
    CUDA_CHECK(cudaGetLastError());
}

/* ══════════════════════════════════════════════════════════════════
 * FP16 Half-Precision & Tensor Core Support
 * ══════════════════════════════════════════════════════════════════ */

void *cuda_malloc_f16(size_t n) {
    __half *ptr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&ptr, n * sizeof(__half)));
    return (void *)ptr;
}

void cuda_free_f16(void *dev_ptr) {
    if (dev_ptr) CUDA_CHECK(cudaFree(dev_ptr));
}

__global__ void k_f32_to_f16(const float *src, __half *dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

void cuda_f32_to_f16(const float *src_f32, void *dst_f16, size_t n) {
    k_f32_to_f16<<<blocks_for(n), THREADS, 0, g_active_stream>>>(
        src_f32, (__half *)dst_f16, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_f16_to_f32(const __half *src, float *dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

void cuda_f16_to_f32(const void *src_f16, float *dst_f32, size_t n) {
    k_f16_to_f32<<<blocks_for(n), THREADS, 0, g_active_stream>>>(
        (__half *)src_f16, dst_f32, n);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_matmul_fp16(const void *a_f16, const void *b_f16, void *out_f16,
                       int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmEx(cublas_handle(),
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, M, K,
                                &alpha,
                                b_f16, CUDA_R_16F, N,
                                a_f16, CUDA_R_16F, K,
                                &beta,
                                out_f16, CUDA_R_16F, N,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
