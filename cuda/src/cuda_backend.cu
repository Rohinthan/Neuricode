/*
 * cuda/src/cuda_backend.cu
 * ────────────────────────────────────────────────────────────────
 * Optimized CUDA kernels + cuBLAS launchers backing cuda_backend.h
 *
 * Provides:
 * 1. cuBLAS GEMM-based Linear Layer Forward & Backward
 * 2. Reusable GPU Memory Pool Workspace (with peak tracking & bounds checking)
 * 3. Warp-Level Shuffle Softmax (__shfl_down_sync) & Large Block Fallback
 * 4. Fused Bias + Activation Kernels (ReLU, GELU, SiLU)
 * 5. FP16 Half-Precision + Tensor Core (cublasGemmEx)
 * 6. VRAM Memory Footprint Diagnostics
 *
 * Compiled with nvcc into build/cuda_backend.o.
 */

#include "../include/cuda_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

/* ── Error Handling Macros ───────────────────────────────────────── */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                        \
        if (_err != cudaSuccess) {                                        \
            fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err));                             \
            exit(1);                                                      \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _st = (call);                                       \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "[cuBLAS Error] %s:%d: status %d\n", __FILE__,  \
                    __LINE__, (int)_st);                                   \
            exit(1);                                                      \
        }                                                                   \
    } while (0)

#define THREADS 256
#define WARP_SIZE 32

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

/* ── Reusable GPU Memory Pool Workspace ─────────────────────────── */
static void  *g_pool_base = NULL;
static size_t g_pool_capacity = 0;
static size_t g_pool_offset = 0;
static size_t g_pool_peak = 0;

void cuda_pool_init(size_t bytes) {
    if (g_pool_base) {
        if (g_pool_capacity >= bytes) {
            g_pool_offset = 0;
            return;
        }
        cuda_free_f32((float *)g_pool_base);
        g_pool_base = NULL;
    }
    CUDA_CHECK(cudaMalloc(&g_pool_base, bytes));
    g_pool_capacity = bytes;
    g_pool_offset = 0;
    g_pool_peak = 0;
}

void *cuda_pool_alloc(size_t bytes) {
    size_t aligned_bytes = (bytes + 255) & ~((size_t)255);
    if (!g_pool_base || g_pool_offset + aligned_bytes > g_pool_capacity) {
        fprintf(stderr, "[GPU Pool Error] Workspace exhausted! Requested %zu bytes, Capacity %zu, Current Offset %zu\n",
                bytes, g_pool_capacity, g_pool_offset);
        exit(1);
    }
    void *ptr = (char *)g_pool_base + g_pool_offset;
    g_pool_offset += aligned_bytes;
    if (g_pool_offset > g_pool_peak) g_pool_peak = g_pool_offset;
    return ptr;
}

void cuda_pool_reset(void) {
    g_pool_offset = 0;
}

void cuda_pool_destroy(void) {
    if (g_pool_base) {
        CUDA_CHECK(cudaFree(g_pool_base));
        g_pool_base = NULL;
        g_pool_capacity = 0;
        g_pool_offset = 0;
        g_pool_peak = 0;
    }
}

void cuda_pool_info(size_t *capacity, size_t *used, size_t *peak) {
    if (capacity) *capacity = g_pool_capacity;
    if (used)     *used     = g_pool_offset;
    if (peak)     *peak     = g_pool_peak;
}

/* ── VRAM Diagnostics ───────────────────────────────────────────── */
size_t estimate_dense_vram_bytes(int batch, int in_f, int out_f) {
    size_t params = (size_t)out_f * in_f + out_f;
    size_t activations = (size_t)batch * (in_f + out_f);
    size_t gradients = params + activations;
    return (params + activations + gradients) * sizeof(float);
}

void cuda_get_vram_info(size_t *free_bytes, size_t *total_bytes) {
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (free_bytes)  *free_bytes  = free_b;
    if (total_bytes) *total_bytes = total_b;
}

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

void cuda_matmul(const float *a, const float *b, float *out, int M, int K, int N) {
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

__global__ void k_fused_bias_add(float *z, const float *bias, int batch, int cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cols;
    if (idx >= total) return;
    int c = idx % cols;
    z[idx] += bias[c];
}

void cuda_fused_bias_add(float *z, const float *bias, int batch, int cols) {
    size_t total = (size_t)batch * cols;
    k_fused_bias_add<<<blocks_for(total), THREADS, 0, g_active_stream>>>(z, bias, batch, cols);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_linear_forward(const float *x, const float *W, const float *bias,
                          float *z, int batch, int in_f, int out_f) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle(),
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              out_f, batch, in_f,
                              &alpha,
                              W, in_f,
                              x, in_f,
                              &beta,
                              z, out_f));
    if (bias) cuda_fused_bias_add(z, bias, batch, out_f);
}

__global__ void k_linear_db_coalesced(const float *dz, float *db, int batch, int out_f) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_f) return;
    float acc = 0.0f;
    for (int b = 0; b < batch; b++) acc += dz[(size_t)b * out_f + o];
    db[o] = acc / (float)batch;
}

void cuda_linear_backward(const float *x, const float *W, const float *dz,
                           float *dW, float *db, float *dX,
                           int batch, int in_f, int out_f) {
    if (dW) {
        const float alpha = 1.0f / (float)batch, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle(),
                                  CUBLAS_OP_N, CUBLAS_OP_T,
                                  in_f, out_f, batch,
                                  &alpha,
                                  x, in_f,
                                  dz, out_f,
                                  &beta,
                                  dW, in_f));
    }
    if (db) {
        k_linear_db_coalesced<<<blocks_for(out_f), THREADS, 0, g_active_stream>>>(
            dz, db, batch, out_f);
        CUDA_CHECK(cudaGetLastError());
    }
    if (dX) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle(),
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  in_f, batch, out_f,
                                  &alpha,
                                  W, in_f,
                                  dz, out_f,
                                  &beta,
                                  dX, in_f));
    }
}

__global__ void k_fused_bias_relu(float *z, const float *bias, int batch, int cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cols;
    if (idx >= total) return;
    int c = idx % cols;
    float v = z[idx] + bias[c];
    z[idx] = v > 0.0f ? v : 0.0f;
}

void cuda_fused_bias_relu(float *z, const float *bias, int batch, int cols) {
    size_t total = (size_t)batch * cols;
    k_fused_bias_relu<<<blocks_for(total), THREADS, 0, g_active_stream>>>(z, bias, batch, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_fused_bias_gelu(float *z, const float *bias, int batch, int cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cols;
    if (idx >= total) return;
    int c = idx % cols;
    float x = z[idx] + bias[c];
    float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    z[idx] = x * cdf;
}

void cuda_fused_bias_gelu(float *z, const float *bias, int batch, int cols) {
    size_t total = (size_t)batch * cols;
    k_fused_bias_gelu<<<blocks_for(total), THREADS, 0, g_active_stream>>>(z, bias, batch, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_fused_bias_silu(float *z, const float *bias, int batch, int cols) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cols;
    if (idx >= total) return;
    int c = idx % cols;
    float x = z[idx] + bias[c];
    z[idx] = x / (1.0f + expf(-x));
}

void cuda_fused_bias_silu(float *z, const float *bias, int batch, int cols) {
    size_t total = (size_t)batch * cols;
    k_fused_bias_silu<<<blocks_for(total), THREADS, 0, g_active_stream>>>(z, bias, batch, cols);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_fused_linear_relu(const float *x, const float *W, const float *bias,
                             float *out, int batch, int in_f, int out_f) {
    cuda_linear_forward(x, W, NULL, out, batch, in_f, out_f);
    if (bias) {
        cuda_fused_bias_relu(out, bias, batch, out_f);
    } else {
        cuda_relu(out, out, (size_t)batch * out_f);
    }
}

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

void cuda_fused_add_bias_relu(const float *a, const float *bias, float *out,
                              int batch, int cols) {
    size_t total = (size_t)batch * cols;
    cuda_memcpy_d2d(out, a, total);
    cuda_fused_bias_relu(out, bias, batch, cols);
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

/* Warp-level shuffle primitives */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

/* Warp-level Softmax (cols <= 1024) */
__global__ void k_softmax_warp(const float * __restrict__ a,
                               float * __restrict__ out, int cols) {
    int row = blockIdx.x * (blockDim.y) + threadIdx.y;
    int lane = threadIdx.x;
    const float *in_row = a + (size_t)row * cols;
    float *out_row = out + (size_t)row * cols;

    float local_max = -FLT_MAX;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        local_max = fmaxf(local_max, in_row[c]);
    }
    float row_max = warp_reduce_max(local_max);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float local_sum = 0.0f;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        float e = expf(in_row[c] - row_max);
        out_row[c] = e;
        local_sum += e;
    }
    float row_sum = warp_reduce_sum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    float inv_sum = 1.0f / row_sum;
    for (int c = lane; c < cols; c += WARP_SIZE) {
        out_row[c] *= inv_sum;
    }
}

/* Block-wide Reduction Softmax Fallback (cols > 1024) */
__global__ void k_softmax_block(const float * __restrict__ a,
                                float * __restrict__ out, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *in_row = a + (size_t)row * cols;
    float *out_row = out + (size_t)row * cols;

    extern __shared__ float s_data[];

    /* 1. Maximum Reduction */
    float local_max = -FLT_MAX;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, in_row[c]);
    }
    s_data[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_data[0];

    /* 2. Exp and Sum Reduction */
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float e = expf(in_row[c] - row_max);
        out_row[c] = e;
        local_sum += e;
    }
    s_data[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / s_data[0];

    /* 3. Normalize */
    for (int c = tid; c < cols; c += blockDim.x) {
        out_row[c] *= inv_sum;
    }
}

void cuda_softmax_rows(const float *a, float *out, int rows, int cols) {
    if (cols <= 1024) {
        dim3 block(WARP_SIZE, 8);
        dim3 grid((rows + 7) / 8);
        k_softmax_warp<<<grid, block, 0, g_active_stream>>>(a, out, cols);
    } else {
        dim3 block(THREADS);
        dim3 grid(rows);
        size_t shmem = THREADS * sizeof(float);
        k_softmax_block<<<grid, block, shmem, g_active_stream>>>(a, out, cols);
    }
    CUDA_CHECK(cudaGetLastError());
}

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
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
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

void cuda_linear_forward_fp16(const void *x_f16, const void *w_f16, const float *bias,
                              void *z_f16, int batch, int in_f, int out_f) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                out_f, batch, in_f,
                                &alpha,
                                w_f16, CUDA_R_16F, in_f,
                                x_f16, CUDA_R_16F, in_f,
                                &beta,
                                z_f16, CUDA_R_16F, out_f,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    (void)bias;
}

void cuda_linear_backward_fp16(const void *x_f16, const void *w_f16, const void *dz_f16,
                               void *dw_f16, float *db_f32, void *dx_f16,
                               int batch, int in_f, int out_f) {
    if (dw_f16) {
        const float alpha = 1.0f / (float)batch, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                    in_f, out_f, batch,
                                    &alpha,
                                    x_f16, CUDA_R_16F, in_f,
                                    dz_f16, CUDA_R_16F, out_f,
                                    &beta,
                                    dw_f16, CUDA_R_16F, in_f,
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    if (dx_f16) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cublas_handle(),
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    in_f, batch, out_f,
                                    &alpha,
                                    w_f16, CUDA_R_16F, in_f,
                                    dz_f16, CUDA_R_16F, out_f,
                                    &beta,
                                    dx_f16, CUDA_R_16F, in_f,
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    (void)db_f32;
}

#else /* CPU Fallback Stubs when USE_CUDA is disabled */

int    cuda_available(void)         { return 0; }
int    cuda_get_device_count(void) { return 0; }
void   cuda_set_device(int id)     { (void)id; }
int    cuda_get_device(void)       { return 0; }
void  *cuda_stream_create(void)    { return NULL; }
void   cuda_stream_destroy(void *s){ (void)s; }
void   cuda_set_stream(void *s)    { (void)s; }
void  *cuda_get_stream(void)       { return NULL; }
void   cuda_enable_tensor_cores(int e) { (void)e; }

void   cuda_pool_init(size_t b)    { (void)b; }
void  *cuda_pool_alloc(size_t b)   { (void)b; return NULL; }
void   cuda_pool_reset(void)       {}
void   cuda_pool_destroy(void)     {}
void   cuda_pool_info(size_t *c, size_t *u, size_t *p) { if (c) *c=0; if (u) *u=0; if (p) *p=0; }

size_t estimate_dense_vram_bytes(int batch, int in_f, int out_f) {
    size_t params = (size_t)out_f * in_f + out_f;
    size_t activations = (size_t)batch * (in_f + out_f);
    size_t gradients = params + activations;
    return (params + activations + gradients) * sizeof(float);
}

void cuda_get_vram_info(size_t *free_bytes, size_t *total_bytes) {
    if (free_bytes)  *free_bytes  = 0;
    if (total_bytes) *total_bytes = 0;
}

float *cuda_malloc_f32(size_t n)   { (void)n; return NULL; }
void   cuda_free_f32(float *p)     { (void)p; }
float *cuda_malloc_host(size_t n)  { (void)n; return NULL; }
void   cuda_free_host(float *p)    { (void)p; }

void   cuda_memcpy_h2d(float *d, const float *s, size_t n) { (void)d; (void)s; (void)n; }
void   cuda_memcpy_d2h(float *d, const float *s, size_t n) { (void)d; (void)s; (void)n; }
void   cuda_memcpy_d2d(float *d, const float *s, size_t n) { (void)d; (void)s; (void)n; }
void   cuda_memcpy_h2d_async(float *d, const float *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; }
void   cuda_memcpy_d2h_async(float *d, const float *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; }
void   cuda_memcpy_d2d_async(float *d, const float *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; }
void   cuda_memset_zero(float *p, size_t n) { (void)p; (void)n; }
void   cuda_sync(void)             {}

void cuda_matmul(const float *a, const float *b, float *out, int M, int K, int N) { (void)a; (void)b; (void)out; (void)M; (void)K; (void)N; }
void cuda_transpose(const float *a, float *out, int r, int c) { (void)a; (void)out; (void)r; (void)c; }
void cuda_linear_forward(const float *x, const float *W, const float *b, float *z, int batch, int i, int o) { (void)x; (void)W; (void)b; (void)z; (void)batch; (void)i; (void)o; }
void cuda_linear_backward(const float *x, const float *W, const float *dz, float *dW, float *db, float *dX, int batch, int i, int o) { (void)x; (void)W; (void)dz; (void)dW; (void)db; (void)dX; (void)batch; (void)i; (void)o; }
void cuda_fused_linear_relu(const float *x, const float *W, const float *b, float *out, int batch, int i, int o) { (void)x; (void)W; (void)b; (void)out; (void)batch; (void)i; (void)o; }
void cuda_fused_bias_add(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_bias_relu(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_bias_gelu(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_bias_silu(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_add_relu(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_fused_mul_relu(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_fused_add_bias_relu(const float *a, const float *b, float *out, int batch, int cols) { (void)a; (void)b; (void)out; (void)batch; (void)cols; }

void cuda_add(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_sub(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_mul(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_scale(const float *a, float s, float *out, size_t n) { (void)a; (void)s; (void)out; (void)n; }
void cuda_add_scalar(const float *a, float s, float *out, size_t n) { (void)a; (void)s; (void)out; (void)n; }
void cuda_fill(float *a, float val, size_t n) { (void)a; (void)val; (void)n; }

void cuda_relu(const float *a, float *out, size_t n) { (void)a; (void)out; (void)n; }
void cuda_relu_grad(const float *a, const float *g, float *out, size_t n) { (void)a; (void)g; (void)out; (void)n; }
void cuda_sigmoid(const float *a, float *out, size_t n) { (void)a; (void)out; (void)n; }
void cuda_sigmoid_grad(const float *s, const float *g, float *out, size_t n) { (void)s; (void)g; (void)out; (void)n; }
void cuda_tanh_f(const float *a, float *out, size_t n) { (void)a; (void)out; (void)n; }
void cuda_tanh_grad(const float *t, const float *g, float *out, size_t n) { (void)t; (void)g; (void)out; (void)n; }
void cuda_softmax_rows(const float *a, float *out, int r, int c) { (void)a; (void)out; (void)r; (void)c; }

void cuda_pad2d(const float *in, float *out, int b, int C, int H, int W, int pad) { (void)in; (void)out; (void)b; (void)C; (void)H; (void)W; (void)pad; }
void cuda_unpad2d(const float *p, float *out, int b, int C, int H, int W, int pad) { (void)p; (void)out; (void)b; (void)C; (void)H; (void)W; (void)pad; }
void cuda_conv2d_forward(const float *in, const float *W, const float *b, float *out, int b_n, int iC, int pH, int pW, int oC, int kH, int kW, int st, int oH, int oW) { (void)in; (void)W; (void)b; (void)out; (void)b_n; (void)iC; (void)pH; (void)pW; (void)oC; (void)kH; (void)kW; (void)st; (void)oH; (void)oW; }
void cuda_conv2d_backward(const float *in, const float *W, const float *g_out, float *dW, float *db, float *dp, int b_n, int iC, int pH, int pW, int oC, int kH, int kW, int st, int oH, int oW) { (void)in; (void)W; (void)g_out; (void)dW; (void)db; (void)dp; (void)b_n; (void)iC; (void)pH; (void)pW; (void)oC; (void)kH; (void)kW; (void)st; (void)oH; (void)oW; }

void cuda_maxpool2d_forward(const float *in, float *out, float *mask, int b, int C, int H, int W, int ps, int st, int oH, int oW) { (void)in; (void)out; (void)mask; (void)b; (void)C; (void)H; (void)W; (void)ps; (void)st; (void)oH; (void)oW; }
void cuda_maxpool2d_backward(const float *go, float *gi, const float *mask, int b, int C, int H, int W, int ps, int st, int oH, int oW) { (void)go; (void)gi; (void)mask; (void)b; (void)C; (void)H; (void)W; (void)ps; (void)st; (void)oH; (void)oW; }

void *cuda_malloc_f16(size_t n) { (void)n; return NULL; }
void  cuda_free_f16(void *p) { (void)p; }
void  cuda_f32_to_f16(const float *s, void *d, size_t n) { (void)s; (void)d; (void)n; }
void  cuda_f16_to_f32(const void *s, float *d, size_t n) { (void)s; (void)d; (void)n; }
void  cuda_matmul_fp16(const void *a, const void *b, void *o, int M, int K, int N) { (void)a; (void)b; (void)o; (void)M; (void)K; (void)N; }
void  cuda_linear_forward_fp16(const void *x, const void *w, const float *b, void *z, int ba, int i, int o) { (void)x; (void)w; (void)b; (void)z; (void)ba; (void)i; (void)o; }
void  cuda_linear_backward_fp16(const void *x, const void *w, const void *dz, void *dw, float *db, void *dx, int ba, int i, int o) { (void)x; (void)w; (void)dz; (void)dw; (void)db; (void)dx; (void)ba; (void)i; (void)o; }

#endif /* USE_CUDA */
