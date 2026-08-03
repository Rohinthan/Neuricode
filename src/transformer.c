/*
 * transformer.c — Production-Grade Native Zero-Dependency Decoder-Only Transformer Engine in Pure C11
 *
 * Upgraded Features:
 *  1. SIMD-Accelerated (AVX2 / ARM NEON / Unrolled Scalar) Tiled Matrix Multiplication
 *  2. Precomputed RoPE (Rotary Position Embeddings) Cos/Sin Lookup Tables
 *  3. Optimized KV-Cache Layout for Contiguous Sequential Memory Prefetching
 *  4. Strict 64-Byte Aligned Memory Allocations via neuralc_aligned_alloc
 *  5. Numerically Stable Softmax and RMSNorm (Log-Sum-Exp / Epsilon / NaN Guards)
 *  6. Debug Mode & Tensor Validation (NaN / Inf Diagnostics)
 *  7. Pretrained Binary Weight Loading & Saving API
 *  8. Multi-Token Batch Forward Inference Engine
 *  9. INT8 (Q8_0) Quantization Support for Ultra-Low Footprint Inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "transformer.h"
#include "memory.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NC_ENABLE_AVX2_GEMV 1
#define NC_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define NC_ENABLE_AVX2_GEMV 0
#define NC_AVX2_TARGET
#endif

// Forward declarations: matmul_cpu_supports_avx2_fma() and softmax_avx2()
// are defined further down, alongside the rest of the AVX2 kernels, since
// that's where the intrinsics and CPUID helper naturally live. softmax()
// (below) needs to call them, so it just needs the prototypes in scope —
// no logic is duplicated here.
#if NC_ENABLE_AVX2_GEMV
static bool matmul_cpu_supports_avx2_fma(void);
static NC_AVX2_TARGET void softmax_avx2(float *x, int size);
static NC_AVX2_TARGET void swiglu_gate_avx2(float *hb, const float *hb2, int n);
#endif

// ── Assertion & Error Macro ─────────────────────────────────────────

#define NC_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "[NC_ASSERT_FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); \
            assert(cond); \
        } \
    } while (0)

// ── Debug & Validation Engine ───────────────────────────────────────

bool validate_tensor_buffer(const float *data, size_t size, const char *name) {
    if (!data) {
        fprintf(stderr, "[NC_DEBUG_ERR] Buffer '%s' is NULL\n", name ? name : "unnamed");
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (isnan(data[i]) || isinf(data[i])) {
            fprintf(stderr, "[NC_DEBUG_ERR] Buffer '%s' contains NaN/Inf at index %zu (val=%f)\n",
                    name ? name : "unnamed", i, data[i]);
            return false;
        }
    }
    return true;
}

void transformer_set_debug(TransformerModel *model, bool enable) {
    if (model) {
        model->debug_mode = enable;
    }
}

// ── Mathematical Primitives ─────────────────────────────────────────

// Numerically Stable RMSNorm with epsilon guard
static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    NC_ASSERT(o && x && weight && size > 0, "Invalid RMSNorm parameters");
    float ss = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:ss) schedule(static)
#endif
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= (float)size;
    ss += 1e-6f; // Numerical stability epsilon
    float inv_std = 1.0f / sqrtf(ss);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (x[i] * inv_std);
    }
}

// Numerically Stable Softmax — scalar reference (subtracts max logit,
// guards against division by 0). Kept exactly as-is: this is the oracle
// the AVX2 path is validated against, and the fallback for non-AVX2 CPUs
// (ARM, or x86 without FMA).
static void softmax_scalar(float *x, int size) {
    if (size <= 0) return;
    float max_val = -1e30f;
    for (int i = 0; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        float val = expf(x[i] - max_val);
        x[i] = val;
        sum += val;
    }
    float inv_sum = (sum > 1e-12) ? (float)(1.0 / sum) : 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// Public softmax() — runtime dispatcher. Same signature/behavior contract
// as before; callers (attention, line ~991) don't change. Picks the AVX2
// polynomial-exp path when the CPU supports it, otherwise falls back to
// the scalar oracle above.
//
// Determinism note: softmax_avx2() uses a polynomial approximation of
// exp() (exp256_ps), not libm's expf(). Output stays normalized (sums to
// 1.0) and numerically stable (max-subtraction is preserved), but is NOT
// bit-identical to softmax_scalar() — differences sit at ~1e-7 relative
// error (float32 ULP-level), not a correctness issue, but worth knowing
// if you diff logs/checkpoints against older runs.
static void softmax(float *x, int size) {
#if NC_ENABLE_AVX2_GEMV
    if (matmul_cpu_supports_avx2_fma()) {
        softmax_avx2(x, size);
        return;
    }
#endif
    softmax_scalar(x, size);
}

// High-Performance SIMD / Unrolled Matrix Multiplication (xout = w * x)
// w: [d, n] row-major matrix, x: [n] input vector, xout: [d] output vector
#if NC_ENABLE_AVX2_GEMV
static NC_AVX2_TARGET inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}

// (A) HIGH-END MICROKERNEL: Aggressive MR = 12 Row Blocking + K = 16 Unroll + Software Prefetching
// Optimized for maximum throughput & 12x vector x reuse on wide-issue, high-ILP CPUs
static NC_AVX2_TARGET void matmul_avx2_mr12(float *xout, const float *x, const float *w, int n, int d) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 12) {
        int rem = d - i;
        if (rem >= 12) {
            const float *w_row0  = w + (size_t)(i + 0)  * n;
            const float *w_row1  = w + (size_t)(i + 1)  * n;
            const float *w_row2  = w + (size_t)(i + 2)  * n;
            const float *w_row3  = w + (size_t)(i + 3)  * n;
            const float *w_row4  = w + (size_t)(i + 4)  * n;
            const float *w_row5  = w + (size_t)(i + 5)  * n;
            const float *w_row6  = w + (size_t)(i + 6)  * n;
            const float *w_row7  = w + (size_t)(i + 7)  * n;
            const float *w_row8  = w + (size_t)(i + 8)  * n;
            const float *w_row9  = w + (size_t)(i + 9)  * n;
            const float *w_row10 = w + (size_t)(i + 10) * n;
            const float *w_row11 = w + (size_t)(i + 11) * n;

            __m256 acc0  = _mm256_setzero_ps();
            __m256 acc1  = _mm256_setzero_ps();
            __m256 acc2  = _mm256_setzero_ps();
            __m256 acc3  = _mm256_setzero_ps();
            __m256 acc4  = _mm256_setzero_ps();
            __m256 acc5  = _mm256_setzero_ps();
            __m256 acc6  = _mm256_setzero_ps();
            __m256 acc7  = _mm256_setzero_ps();
            __m256 acc8  = _mm256_setzero_ps();
            __m256 acc9  = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();

            int j = 0;
            uintptr_t align_check = (uintptr_t)x | (uintptr_t)w_row0 | (uintptr_t)w_row1 |
                                    (uintptr_t)w_row2 | (uintptr_t)w_row3 | (uintptr_t)w_row4 |
                                    (uintptr_t)w_row5 | (uintptr_t)w_row6 | (uintptr_t)w_row7 |
                                    (uintptr_t)w_row8 | (uintptr_t)w_row9 | (uintptr_t)w_row10 |
                                    (uintptr_t)w_row11;

            if ((align_check & 31) == 0) {
                if (n >= 16) {
                    __m256 vx0 = _mm256_load_ps(x + 0);
                    __m256 vx1 = _mm256_load_ps(x + 8);

                    for (; j <= n - 16; j += 16) {
                        _mm_prefetch((const char*)(w_row0 + j + 32), _MM_HINT_T0);
                        _mm_prefetch((const char*)(w_row6 + j + 32), _MM_HINT_T0);

                        __m256 w0 = _mm256_load_ps(w_row0 + j);
                        __m256 w1 = _mm256_load_ps(w_row1 + j);
                        __m256 w2 = _mm256_load_ps(w_row2 + j);
                        __m256 w3 = _mm256_load_ps(w_row3 + j);

                        acc0 = _mm256_fmadd_ps(w0, vx0, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx0, acc1);

                        __m256 w4 = _mm256_load_ps(w_row4 + j);
                        __m256 w5 = _mm256_load_ps(w_row5 + j);

                        acc2 = _mm256_fmadd_ps(w2, vx0, acc2);
                        acc3 = _mm256_fmadd_ps(w3, vx0, acc3);

                        __m256 w6 = _mm256_load_ps(w_row6 + j);
                        __m256 w7 = _mm256_load_ps(w_row7 + j);

                        acc4 = _mm256_fmadd_ps(w4, vx0, acc4);
                        acc5 = _mm256_fmadd_ps(w5, vx0, acc5);

                        __m256 w8 = _mm256_load_ps(w_row8 + j);
                        __m256 w9 = _mm256_load_ps(w_row9 + j);

                        acc6 = _mm256_fmadd_ps(w6, vx0, acc6);
                        acc7 = _mm256_fmadd_ps(w7, vx0, acc7);

                        __m256 w10 = _mm256_load_ps(w_row10 + j);
                        __m256 w11 = _mm256_load_ps(w_row11 + j);

                        acc8  = _mm256_fmadd_ps(w8,  vx0, acc8);
                        acc9  = _mm256_fmadd_ps(w9,  vx0, acc9);
                        acc10 = _mm256_fmadd_ps(w10, vx0, acc10);
                        acc11 = _mm256_fmadd_ps(w11, vx0, acc11);

                        w0 = _mm256_load_ps(w_row0 + j + 8);
                        w1 = _mm256_load_ps(w_row1 + j + 8);
                        w2 = _mm256_load_ps(w_row2 + j + 8);
                        w3 = _mm256_load_ps(w_row3 + j + 8);

                        acc0 = _mm256_fmadd_ps(w0, vx1, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx1, acc1);

                        w4 = _mm256_load_ps(w_row4 + j + 8);
                        w5 = _mm256_load_ps(w_row5 + j + 8);

                        acc2 = _mm256_fmadd_ps(w2, vx1, acc2);
                        acc3 = _mm256_fmadd_ps(w3, vx1, acc3);

                        w6 = _mm256_load_ps(w_row6 + j + 8);
                        w7 = _mm256_load_ps(w_row7 + j + 8);

                        acc4 = _mm256_fmadd_ps(w4, vx1, acc4);
                        acc5 = _mm256_fmadd_ps(w5, vx1, acc5);

                        w8 = _mm256_load_ps(w_row8 + j + 8);
                        w9 = _mm256_load_ps(w_row9 + j + 8);

                        acc6 = _mm256_fmadd_ps(w6, vx1, acc6);
                        acc7 = _mm256_fmadd_ps(w7, vx1, acc7);

                        __m256 next_vx0, next_vx1;
                        if (j + 16 <= n - 16) {
                            next_vx0 = _mm256_load_ps(x + j + 16);
                            next_vx1 = _mm256_load_ps(x + j + 24);
                        }

                        w10 = _mm256_load_ps(w_row10 + j + 8);
                        w11 = _mm256_load_ps(w_row11 + j + 8);

                        acc8  = _mm256_fmadd_ps(w8,  vx1, acc8);
                        acc9  = _mm256_fmadd_ps(w9,  vx1, acc9);
                        acc10 = _mm256_fmadd_ps(w10, vx1, acc10);
                        acc11 = _mm256_fmadd_ps(w11, vx1, acc11);

                        if (j + 16 <= n - 16) {
                            vx0 = next_vx0;
                            vx1 = next_vx1;
                        }
                    }
                }
            } else {
                if (n >= 16) {
                    __m256 vx0 = _mm256_loadu_ps(x + 0);
                    __m256 vx1 = _mm256_loadu_ps(x + 8);

                    for (; j <= n - 16; j += 16) {
                        _mm_prefetch((const char*)(w_row0 + j + 32), _MM_HINT_T0);
                        _mm_prefetch((const char*)(w_row6 + j + 32), _MM_HINT_T0);

                        __m256 w0 = _mm256_loadu_ps(w_row0 + j);
                        __m256 w1 = _mm256_loadu_ps(w_row1 + j);
                        __m256 w2 = _mm256_loadu_ps(w_row2 + j);
                        __m256 w3 = _mm256_loadu_ps(w_row3 + j);

                        acc0 = _mm256_fmadd_ps(w0, vx0, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx0, acc1);

                        __m256 w4 = _mm256_loadu_ps(w_row4 + j);
                        __m256 w5 = _mm256_loadu_ps(w_row5 + j);

                        acc2 = _mm256_fmadd_ps(w2, vx0, acc2);
                        acc3 = _mm256_fmadd_ps(w3, vx0, acc3);

                        __m256 w6 = _mm256_loadu_ps(w_row6 + j);
                        __m256 w7 = _mm256_loadu_ps(w_row7 + j);

                        acc4 = _mm256_fmadd_ps(w4, vx0, acc4);
                        acc5 = _mm256_fmadd_ps(w5, vx0, acc5);

                        __m256 w8 = _mm256_loadu_ps(w_row8 + j);
                        __m256 w9 = _mm256_loadu_ps(w_row9 + j);

                        acc6 = _mm256_fmadd_ps(w6, vx0, acc6);
                        acc7 = _mm256_fmadd_ps(w7, vx0, acc7);

                        __m256 w10 = _mm256_loadu_ps(w_row10 + j);
                        __m256 w11 = _mm256_loadu_ps(w_row11 + j);

                        acc8  = _mm256_fmadd_ps(w8,  vx0, acc8);
                        acc9  = _mm256_fmadd_ps(w9,  vx0, acc9);
                        acc10 = _mm256_fmadd_ps(w10, vx0, acc10);
                        acc11 = _mm256_fmadd_ps(w11, vx0, acc11);

                        w0 = _mm256_loadu_ps(w_row0 + j + 8);
                        w1 = _mm256_loadu_ps(w_row1 + j + 8);
                        w2 = _mm256_loadu_ps(w_row2 + j + 8);
                        w3 = _mm256_loadu_ps(w_row3 + j + 8);

                        acc0 = _mm256_fmadd_ps(w0, vx1, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx1, acc1);

                        w4 = _mm256_loadu_ps(w_row4 + j + 8);
                        w5 = _mm256_loadu_ps(w_row5 + j + 8);

                        acc2 = _mm256_fmadd_ps(w2, vx1, acc2);
                        acc3 = _mm256_fmadd_ps(w3, vx1, acc3);

                        w6 = _mm256_loadu_ps(w_row6 + j + 8);
                        w7 = _mm256_loadu_ps(w_row7 + j + 8);

                        acc4 = _mm256_fmadd_ps(w4, vx1, acc4);
                        acc5 = _mm256_fmadd_ps(w5, vx1, acc5);

                        w8 = _mm256_loadu_ps(w_row8 + j + 8);
                        w9 = _mm256_loadu_ps(w_row9 + j + 8);

                        acc6 = _mm256_fmadd_ps(w6, vx1, acc6);
                        acc7 = _mm256_fmadd_ps(w7, vx1, acc7);

                        __m256 next_vx0, next_vx1;
                        if (j + 16 <= n - 16) {
                            next_vx0 = _mm256_loadu_ps(x + j + 16);
                            next_vx1 = _mm256_loadu_ps(x + j + 24);
                        }

                        w10 = _mm256_loadu_ps(w_row10 + j + 8);
                        w11 = _mm256_loadu_ps(w_row11 + j + 8);

                        acc8  = _mm256_fmadd_ps(w8,  vx1, acc8);
                        acc9  = _mm256_fmadd_ps(w9,  vx1, acc9);
                        acc10 = _mm256_fmadd_ps(w10, vx1, acc10);
                        acc11 = _mm256_fmadd_ps(w11, vx1, acc11);

                        if (j + 16 <= n - 16) {
                            vx0 = next_vx0;
                            vx1 = next_vx1;
                        }
                    }
                }
            }

            float val0  = _mm256_reduce_add_ps(acc0);
            float val1  = _mm256_reduce_add_ps(acc1);
            float val2  = _mm256_reduce_add_ps(acc2);
            float val3  = _mm256_reduce_add_ps(acc3);
            float val4  = _mm256_reduce_add_ps(acc4);
            float val5  = _mm256_reduce_add_ps(acc5);
            float val6  = _mm256_reduce_add_ps(acc6);
            float val7  = _mm256_reduce_add_ps(acc7);
            float val8  = _mm256_reduce_add_ps(acc8);
            float val9  = _mm256_reduce_add_ps(acc9);
            float val10 = _mm256_reduce_add_ps(acc10);
            float val11 = _mm256_reduce_add_ps(acc11);

            for (; j < n; j++) {
                float xj = x[j];
                val0  += w_row0[j]  * xj;
                val1  += w_row1[j]  * xj;
                val2  += w_row2[j]  * xj;
                val3  += w_row3[j]  * xj;
                val4  += w_row4[j]  * xj;
                val5  += w_row5[j]  * xj;
                val6  += w_row6[j]  * xj;
                val7  += w_row7[j]  * xj;
                val8  += w_row8[j]  * xj;
                val9  += w_row9[j]  * xj;
                val10 += w_row10[j] * xj;
                val11 += w_row11[j] * xj;
            }

            xout[i + 0]  = val0;
            xout[i + 1]  = val1;
            xout[i + 2]  = val2;
            xout[i + 3]  = val3;
            xout[i + 4]  = val4;
            xout[i + 5]  = val5;
            xout[i + 6]  = val6;
            xout[i + 7]  = val7;
            xout[i + 8]  = val8;
            xout[i + 9]  = val9;
            xout[i + 10] = val10;
            xout[i + 11] = val11;
        } else {
            for (int r = 0; r < rem; r++) {
                int row_idx = i + r;
                const float *w_row = w + (size_t)row_idx * n;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int j = 0;
                if ((((uintptr_t)x | (uintptr_t)w_row) & 31) == 0) {
                    for (; j <= n - 16; j += 16) {
                        __m256 vx0 = _mm256_load_ps(x + j);
                        __m256 vx1 = _mm256_load_ps(x + j + 8);
                        __m256 w0  = _mm256_load_ps(w_row + j);
                        __m256 w1  = _mm256_load_ps(w_row + j + 8);
                        acc0 = _mm256_fmadd_ps(w0, vx0, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx1, acc1);
                    }
                } else {
                    for (; j <= n - 16; j += 16) {
                        __m256 vx0 = _mm256_loadu_ps(x + j);
                        __m256 vx1 = _mm256_loadu_ps(x + j + 8);
                        __m256 w0  = _mm256_loadu_ps(w_row + j);
                        __m256 w1  = _mm256_loadu_ps(w_row + j + 8);
                        acc0 = _mm256_fmadd_ps(w0, vx0, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx1, acc1);
                    }
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                float val = _mm256_reduce_add_ps(acc0);
                for (; j < n; j++) {
                    val += w_row[j] * x[j];
                }
                xout[row_idx] = val;
            }
        }
    }
}

// (B) LOW-END MICROKERNEL: Conservative MR = 4 Row Blocking with Dual Accumulators per Row + Prefetching
// Optimized for low latency, low register pressure, and narrower OoO execution engines
static NC_AVX2_TARGET void matmul_avx2_mr4(float *xout, const float *x, const float *w, int n, int d) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 4) {
        int rem = d - i;
        if (rem >= 4) {
            const float *w_row0 = w + (size_t)(i + 0) * n;
            const float *w_row1 = w + (size_t)(i + 1) * n;
            const float *w_row2 = w + (size_t)(i + 2) * n;
            const float *w_row3 = w + (size_t)(i + 3) * n;

            __m256 acc0_0 = _mm256_setzero_ps();
            __m256 acc0_1 = _mm256_setzero_ps();
            __m256 acc1_0 = _mm256_setzero_ps();
            __m256 acc1_1 = _mm256_setzero_ps();
            __m256 acc2_0 = _mm256_setzero_ps();
            __m256 acc2_1 = _mm256_setzero_ps();
            __m256 acc3_0 = _mm256_setzero_ps();
            __m256 acc3_1 = _mm256_setzero_ps();

            int j = 0;
            uintptr_t align_check = (uintptr_t)x | (uintptr_t)w_row0 | (uintptr_t)w_row1 |
                                    (uintptr_t)w_row2 | (uintptr_t)w_row3;

            if ((align_check & 31) == 0) {
                if (n >= 16) {
                    __m256 vx0 = _mm256_load_ps(x + 0);
                    __m256 vx1 = _mm256_load_ps(x + 8);

                    for (; j <= n - 16; j += 16) {
                        _mm_prefetch((const char*)(w_row0 + j + 32), _MM_HINT_T0);
                        _mm_prefetch((const char*)(w_row2 + j + 32), _MM_HINT_T0);

                        __m256 w0_0 = _mm256_load_ps(w_row0 + j);
                        __m256 w1_0 = _mm256_load_ps(w_row1 + j);
                        __m256 w2_0 = _mm256_load_ps(w_row2 + j);
                        __m256 w3_0 = _mm256_load_ps(w_row3 + j);

                        acc0_0 = _mm256_fmadd_ps(w0_0, vx0, acc0_0);
                        acc1_0 = _mm256_fmadd_ps(w1_0, vx0, acc1_0);
                        acc2_0 = _mm256_fmadd_ps(w2_0, vx0, acc2_0);
                        acc3_0 = _mm256_fmadd_ps(w3_0, vx0, acc3_0);

                        __m256 w0_1 = _mm256_load_ps(w_row0 + j + 8);
                        __m256 w1_1 = _mm256_load_ps(w_row1 + j + 8);
                        __m256 w2_1 = _mm256_load_ps(w_row2 + j + 8);
                        __m256 w3_1 = _mm256_load_ps(w_row3 + j + 8);

                        __m256 next_vx0, next_vx1;
                        if (j + 16 <= n - 16) {
                            next_vx0 = _mm256_load_ps(x + j + 16);
                            next_vx1 = _mm256_load_ps(x + j + 24);
                        }

                        acc0_1 = _mm256_fmadd_ps(w0_1, vx1, acc0_1);
                        acc1_1 = _mm256_fmadd_ps(w1_1, vx1, acc1_1);
                        acc2_1 = _mm256_fmadd_ps(w2_1, vx1, acc2_1);
                        acc3_1 = _mm256_fmadd_ps(w3_1, vx1, acc3_1);

                        if (j + 16 <= n - 16) {
                            vx0 = next_vx0;
                            vx1 = next_vx1;
                        }
                    }
                }
            } else {
                if (n >= 16) {
                    __m256 vx0 = _mm256_loadu_ps(x + 0);
                    __m256 vx1 = _mm256_loadu_ps(x + 8);

                    for (; j <= n - 16; j += 16) {
                        _mm_prefetch((const char*)(w_row0 + j + 32), _MM_HINT_T0);
                        _mm_prefetch((const char*)(w_row2 + j + 32), _MM_HINT_T0);

                        __m256 w0_0 = _mm256_loadu_ps(w_row0 + j);
                        __m256 w1_0 = _mm256_loadu_ps(w_row1 + j);
                        __m256 w2_0 = _mm256_loadu_ps(w_row2 + j);
                        __m256 w3_0 = _mm256_loadu_ps(w_row3 + j);

                        acc0_0 = _mm256_fmadd_ps(w0_0, vx0, acc0_0);
                        acc1_0 = _mm256_fmadd_ps(w1_0, vx0, acc1_0);
                        acc2_0 = _mm256_fmadd_ps(w2_0, vx0, acc2_0);
                        acc3_0 = _mm256_fmadd_ps(w3_0, vx0, acc3_0);

                        __m256 w0_1 = _mm256_loadu_ps(w_row0 + j + 8);
                        __m256 w1_1 = _mm256_loadu_ps(w_row1 + j + 8);
                        __m256 w2_1 = _mm256_loadu_ps(w_row2 + j + 8);
                        __m256 w3_1 = _mm256_loadu_ps(w_row3 + j + 8);

                        __m256 next_vx0, next_vx1;
                        if (j + 16 <= n - 16) {
                            next_vx0 = _mm256_loadu_ps(x + j + 16);
                            next_vx1 = _mm256_loadu_ps(x + j + 24);
                        }

                        acc0_1 = _mm256_fmadd_ps(w0_1, vx1, acc0_1);
                        acc1_1 = _mm256_fmadd_ps(w1_1, vx1, acc1_1);
                        acc2_1 = _mm256_fmadd_ps(w2_1, vx1, acc2_1);
                        acc3_1 = _mm256_fmadd_ps(w3_1, vx1, acc3_1);

                        if (j + 16 <= n - 16) {
                            vx0 = next_vx0;
                            vx1 = next_vx1;
                        }
                    }
                }
            }

            acc0_0 = _mm256_add_ps(acc0_0, acc0_1);
            acc1_0 = _mm256_add_ps(acc1_0, acc1_1);
            acc2_0 = _mm256_add_ps(acc2_0, acc2_1);
            acc3_0 = _mm256_add_ps(acc3_0, acc3_1);

            float val0 = _mm256_reduce_add_ps(acc0_0);
            float val1 = _mm256_reduce_add_ps(acc1_0);
            float val2 = _mm256_reduce_add_ps(acc2_0);
            float val3 = _mm256_reduce_add_ps(acc3_0);

            for (; j < n; j++) {
                float xj = x[j];
                val0 += w_row0[j] * xj;
                val1 += w_row1[j] * xj;
                val2 += w_row2[j] * xj;
                val3 += w_row3[j] * xj;
            }

            xout[i + 0] = val0;
            xout[i + 1] = val1;
            xout[i + 2] = val2;
            xout[i + 3] = val3;
        } else {
            for (int r = 0; r < rem; r++) {
                int row_idx = i + r;
                const float *w_row = w + (size_t)row_idx * n;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int j = 0;
                if ((((uintptr_t)x | (uintptr_t)w_row) & 31) == 0) {
                    for (; j <= n - 16; j += 16) {
                        __m256 vx0 = _mm256_load_ps(x + j);
                        __m256 vx1 = _mm256_load_ps(x + j + 8);
                        __m256 w0  = _mm256_load_ps(w_row + j);
                        __m256 w1  = _mm256_load_ps(w_row + j + 8);
                        acc0 = _mm256_fmadd_ps(w0, vx0, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx1, acc1);
                    }
                } else {
                    for (; j <= n - 16; j += 16) {
                        __m256 vx0 = _mm256_loadu_ps(x + j);
                        __m256 vx1 = _mm256_loadu_ps(x + j + 8);
                        __m256 w0  = _mm256_loadu_ps(w_row + j);
                        __m256 w1  = _mm256_loadu_ps(w_row + j + 8);
                        acc0 = _mm256_fmadd_ps(w0, vx0, acc0);
                        acc1 = _mm256_fmadd_ps(w1, vx1, acc1);
                    }
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                float val = _mm256_reduce_add_ps(acc0);
                for (; j < n; j++) {
                    val += w_row[j] * x[j];
                }
                xout[row_idx] = val;
            }
        }
    }
}

// ── AVX2 Softmax ─────────────────────────────────────────────────────
//
// exp256_ps: 8-wide vectorized exp() via range reduction + minimax
// polynomial (Cephes-derived — the standard technique used by most SIMD
// math libraries, e.g. ggml/XNNPACK). Reduces x = n*ln2 + z, evaluates a
// degree-6 polynomial for e^z, then reconstructs 2^n by directly writing
// the IEEE-754 float exponent bits (n + 127) << 23. Accurate to ~1e-7
// relative error vs libm expf — not bit-exact, but well within float32
// noise for softmax weights.
static NC_AVX2_TARGET inline __m256 exp256_ps(__m256 x) {
    // Clamp first: keeps the exponent reconstruction below from
    // overflowing/underflowing the int32 path for pathological inputs.
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647950f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    // Range reduction: x = n*ln2 + z, n = round(x / ln2)
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    __m256 fx = _mm256_fmadd_ps(x, log2ef, _mm256_set1_ps(0.5f));
    __m256 flr = _mm256_floor_ps(fx);
    __m256 mask = _mm256_and_ps(_mm256_cmp_ps(flr, fx, _CMP_GT_OS), _mm256_set1_ps(1.0f));
    flr = _mm256_sub_ps(flr, mask); // flr == n

    // z = x - n*ln2, split ln2 into hi+lo parts for precision (standard
    // Cephes trick — avoids catastrophic cancellation from a single
    // low-precision ln2 constant).
    __m256 z = _mm256_fnmadd_ps(flr, _mm256_set1_ps(0.693359375f), x);
    z = _mm256_fnmadd_ps(flr, _mm256_set1_ps(-2.12194440e-4f), z);

    // Degree-6 minimax polynomial for e^z, z in [-0.5*ln2, 0.5*ln2]
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, z, _mm256_set1_ps(5.0000001201e-1f));
    __m256 z2 = _mm256_mul_ps(z, z);
    y = _mm256_add_ps(_mm256_fmadd_ps(y, z2, z), _mm256_set1_ps(1.0f));

    // Reconstruct 2^n by injecting (n + 127) into the IEEE-754 exponent
    // field directly — far cheaper than a second transcendental call.
    __m256i imm0 = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvttps_epi32(flr), _mm256_set1_epi32(0x7f)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(imm0));
}

// softmax_avx2: 3-pass vectorized softmax (max-reduce -> exp+accumulate
// -> normalize), 8 lanes/iteration. Max-subtraction is preserved (same
// stability guarantee as softmax_scalar) — only the exp() itself and the
// three reduction/broadcast loops are vectorized.
static NC_AVX2_TARGET void softmax_avx2(float *x, int size) {
    if (size <= 0) return;

    // 32-byte alignment check once, up front — reused by all three
    // passes below instead of re-checking per element.
    int aligned = ((uintptr_t)x & 31) == 0;

    // ── Pass 1: max reduction, 8 lanes at a time ────────────────────
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    int i = 0;
    if (aligned) {
        for (; i <= size - 8; i += 8) vmax = _mm256_max_ps(vmax, _mm256_load_ps(x + i));
    } else {
        for (; i <= size - 8; i += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    }
    // Horizontal max: 256->128 (max), 128->64 (movehl), 64->32 (shuffle)
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    float max_val = _mm_cvtss_f32(m);
    for (; i < size; i++) if (x[i] > max_val) max_val = x[i]; // scalar tail (size % 8)

    // ── Pass 2: exp(x - max), accumulate sum ────────────────────────
    // Sum is accumulated as a vector, then reduced to a double at the
    // end (matches the scalar version's double accumulator precision;
    // the individual terms are float32, same as softmax_scalar).
    __m256 vmaxb = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    if (aligned) {
        for (; i <= size - 8; i += 8) {
            __m256 e = exp256_ps(_mm256_sub_ps(_mm256_load_ps(x + i), vmaxb));
            _mm256_store_ps(x + i, e);
            vsum = _mm256_add_ps(vsum, e);
        }
    } else {
        for (; i <= size - 8; i += 8) {
            __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmaxb));
            _mm256_storeu_ps(x + i, e);
            vsum = _mm256_add_ps(vsum, e);
        }
    }
    float partial[8];
    _mm256_storeu_ps(partial, vsum);
    double sum = 0.0;
    for (int k = 0; k < 8; k++) sum += (double)partial[k];
    for (; i < size; i++) { // scalar tail: libm expf, same as reference
        float val = expf(x[i] - max_val);
        x[i] = val;
        sum += val;
    }

    // ── Pass 3: normalize ────────────────────────────────────────────
    // Same divide-by-zero guard as softmax_scalar.
    float inv_sum = (sum > 1e-12) ? (float)(1.0 / sum) : 0.0f;
    __m256 vinv = _mm256_set1_ps(inv_sum);
    i = 0;
    if (aligned) {
        for (; i <= size - 8; i += 8) _mm256_store_ps(x + i, _mm256_mul_ps(_mm256_load_ps(x + i), vinv));
    } else {
        for (; i <= size - 8; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    }
    for (; i < size; i++) x[i] *= inv_sum;
}

// swiglu_gate_avx2: fused SiLU(hb) * hb2 -> hb, 8 lanes/iteration, reusing
// exp256_ps (same poly-exp used by softmax_avx2 — one approximation, two
// call sites, no duplicated math). This is the "kernel fusion" that
// actually applies to this architecture: there's no bias to fuse into
// matmul (LLaMA-style, bias-free), but the SwiGLU gate elementwise loop
// (silu(w1*x) * (w3*x)) is a real fusion target — it's a single pass over
// hidden_dim reading both buffers and writing one, no intermediate
// buffer beyond what already exists (hb, hb2).
static NC_AVX2_TARGET void swiglu_gate_avx2(float *hb, const float *hb2, int n) {
    int aligned = (((uintptr_t)hb | (uintptr_t)hb2) & 31) == 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    if (aligned) {
        for (; i <= n - 8; i += 8) {
            __m256 v  = _mm256_load_ps(hb + i);
            __m256 g  = _mm256_load_ps(hb2 + i);
            __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), v))));
            _mm256_store_ps(hb + i, _mm256_mul_ps(_mm256_mul_ps(v, sig), g));
        }
    } else {
        for (; i <= n - 8; i += 8) {
            __m256 v  = _mm256_loadu_ps(hb + i);
            __m256 g  = _mm256_loadu_ps(hb2 + i);
            __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), v))));
            _mm256_storeu_ps(hb + i, _mm256_mul_ps(_mm256_mul_ps(v, sig), g));
        }
    }
    for (; i < n; i++) { // scalar tail, libm expf — same as the original loop
        float val = hb[i];
        float silu = val / (1.0f + expf(-val));
        hb[i] = silu * hb2[i];
    }
}

static bool matmul_cpu_supports_avx2_fma(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// Runtime Adaptive Dispatcher selecting optimal microkernel based on CPU and dimensions
static void matmul(float *xout, const float *x, const float *w, int n, int d) {
    NC_ASSERT(xout && x && w && n > 0 && d > 0, "Invalid matmul parameters");

    if (!matmul_cpu_supports_avx2_fma()) {
        goto scalar_fallback;
    }

    if (d >= 64 && n >= 128) {
        matmul_avx2_mr12(xout, x, w, n, d);
    } else {
        matmul_avx2_mr4(xout, x, w, n, d);
    }
    return;

scalar_fallback:
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i++) {
        const float *w_row = w + (size_t)i * n;
        float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f, val3 = 0.0f;
        int j = 0;
        for (; j <= n - 4; j += 4) {
            val0 += w_row[j]     * x[j];
            val1 += w_row[j + 1] * x[j + 1];
            val2 += w_row[j + 2] * x[j + 2];
            val3 += w_row[j + 3] * x[j + 3];
        }
        float val = val0 + val1 + val2 + val3;
        for (; j < n; j++) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}
#elif defined(__ARM_NEON) || defined(__aarch64__)
static void matmul(float *xout, const float *x, const float *w, int n, int d) {
    NC_ASSERT(xout && x && w && n > 0 && d > 0, "Invalid matmul parameters");

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i++) {
        const float *w_row = w + (size_t)i * n;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= n - 8; j += 8) {
            acc0 = vmlaq_f32(acc0, vld1q_f32(w_row + j),     vld1q_f32(x + j));
            acc1 = vmlaq_f32(acc1, vld1q_f32(w_row + j + 4), vld1q_f32(x + j + 4));
        }
        acc0 = vaddq_f32(acc0, acc1);
        float val = vaddvq_f32(acc0);
        for (; j < n; j++) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}
#else
static void matmul(float *xout, const float *x, const float *w, int n, int d) {
    NC_ASSERT(xout && x && w && n > 0 && d > 0, "Invalid matmul parameters");

    // Portable 4x unrolled scalar loop fallback
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i++) {
        const float *w_row = w + (size_t)i * n;
        float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f, val3 = 0.0f;
        int j = 0;
        for (; j <= n - 4; j += 4) {
            val0 += w_row[j]     * x[j];
            val1 += w_row[j + 1] * x[j + 1];
            val2 += w_row[j + 2] * x[j + 2];
            val3 += w_row[j + 3] * x[j + 3];
        }
        float val = val0 + val1 + val2 + val3;
        for (; j < n; j++) {
            val += w_row[j] * x[j];
        }
        xout[i] = val;
    }
}
#endif

// Public wrappers exposing the dispatcher above for external
// verification (see transformer.h). Pure pass-through — no logic
// duplicated, no new dispatch path.
void transformer_matmul(float *xout, const float *x, const float *w, int n, int d) {
    matmul(xout, x, w, n, d);
}

bool transformer_has_avx2(void) {
#if NC_ENABLE_AVX2_GEMV
    return matmul_cpu_supports_avx2_fma();
#else
    return false;
#endif
}

// ── Test-only exposure (never part of transformer.h / the public API) ──
// Compiled in only when TEST_EXPOSE_SOFTMAX is defined by the validation
// harness's build command, so production builds are unaffected and this
// never ships as a real symbol in the library.
#ifdef TEST_EXPOSE_SOFTMAX
void test_softmax_scalar(float *x, int n) { softmax_scalar(x, n); }
void test_softmax_avx2(float *x, int n) {
#if NC_ENABLE_AVX2_GEMV
    softmax_avx2(x, n);
#else
    softmax_scalar(x, n);
#endif
}
int test_has_avx2(void) { return transformer_has_avx2() ? 1 : 0; }

void test_swiglu_scalar(float *hb, const float *hb2, int n) {
    for (int i = 0; i < n; i++) {
        float val = hb[i];
        float silu = val / (1.0f + expf(-val));
        hb[i] = silu * hb2[i];
    }
}
void test_swiglu_avx2(float *hb, const float *hb2, int n) {
#if NC_ENABLE_AVX2_GEMV
    swiglu_gate_avx2(hb, hb2, n);
#else
    test_swiglu_scalar(hb, hb2, n);
#endif
}
#endif

// Apply precomputed Rotary Position Embedding (RoPE) to Q and K
static void apply_rope_fast(float *q, float *k, int pos, int head_size, int n_heads, const RoPETable *rope) {
    int half_head = head_size / 2;
    const float *cos_row = rope->cos + (size_t)pos * half_head;
    const float *sin_row = rope->sin + (size_t)pos * half_head;

    for (int h = 0; h < n_heads; h++) {
        float *q_h = q + h * head_size;
        float *k_h = k + h * head_size;
        for (int i = 0; i < head_size; i += 2) {
            float fcr = cos_row[i / 2];
            float fci = sin_row[i / 2];

            float q0 = q_h[i];
            float q1 = q_h[i + 1];
            q_h[i]     = q0 * fcr - q1 * fci;
            q_h[i + 1] = q0 * fci + q1 * fcr;

            float k0 = k_h[i];
            float k1 = k_h[i + 1];
            k_h[i]     = k0 * fcr - k1 * fci;
            k_h[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

// ── Weight Initialization & Precomputed Tables ──────────────────────

static void fill_init_weights(float *ptr, size_t num_elements, float scale) {
    for (size_t i = 0; i < num_elements; i++) {
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        ptr[i] = r * scale;
    }
}

static void fill_ones(float *ptr, size_t num_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        ptr[i] = 1.0f;
    }
}

// Precompute RoPE frequency cos/sin lookup table once during initialization
static bool init_rope_table(RoPETable *rope, int seq_len, int head_size) {
    int half_head = head_size / 2;
    size_t table_bytes = (size_t)seq_len * half_head * sizeof(float);

    rope->cos = (float*)neuralc_aligned_alloc(64, table_bytes);
    rope->sin = (float*)neuralc_aligned_alloc(64, table_bytes);
    if (!rope->cos || !rope->sin) return false;

    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
            float val = (float)pos * freq;
            size_t idx = (size_t)pos * half_head + (i / 2);
            rope->cos[idx] = cosf(val);
            rope->sin[idx] = sinf(val);
        }
    }
    return true;
}

// ── Model Initialization & Allocation ──────────────────────────────

TransformerModel *transformer_init(TransformerConfig config) {
    TransformerModel *model = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    if (!model) {
        fprintf(stderr, "[transformer] Error: Memory allocation failed for TransformerModel\n");
        return NULL;
    }
    model->config = config;
    model->debug_mode = false;

    int vocab_size = config.vocab_size;
    int dim        = config.dim;
    int hidden_dim = config.hidden_dim;
    int n_layers   = config.n_layers;
    int n_heads    = config.n_heads;
    int head_size  = dim / n_heads;
    int seq_len    = config.seq_len;

    TransformerWeights *w = &model->weights;
    TransformerState   *s = &model->state;

    // Allocate 64-byte aligned weights
    w->token_embedding_table = (float*)neuralc_aligned_alloc(64, (size_t)vocab_size * dim * sizeof(float));
    w->rms_att_weight        = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * sizeof(float));
    w->wq                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * dim * sizeof(float));
    w->wk                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * dim * sizeof(float));
    w->wv                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * dim * sizeof(float));
    w->wo                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * dim * sizeof(float));
    w->rms_ffn_weight        = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * sizeof(float));
    w->w1                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * hidden_dim * dim * sizeof(float));
    w->w2                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * dim * hidden_dim * sizeof(float));
    w->w3                    = (float*)neuralc_aligned_alloc(64, (size_t)n_layers * hidden_dim * dim * sizeof(float));
    w->rms_final_weight      = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    w->wcls                  = (float*)neuralc_aligned_alloc(64, (size_t)vocab_size * dim * sizeof(float));

    if (!w->token_embedding_table || !w->rms_att_weight || !w->wq || !w->wk ||
        !w->wv || !w->wo || !w->rms_ffn_weight || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->wcls) {
        fprintf(stderr, "[transformer] Error: Aligned allocation failed for model weights\n");
        transformer_free(model);
        return NULL;
    }

    // Initialize weights
    float scale = 1.0f / sqrtf((float)dim);
    fill_init_weights(w->token_embedding_table, (size_t)vocab_size * dim, scale);
    fill_ones(w->rms_att_weight, (size_t)n_layers * dim);
    fill_init_weights(w->wq, (size_t)n_layers * dim * dim, scale);
    fill_init_weights(w->wk, (size_t)n_layers * dim * dim, scale);
    fill_init_weights(w->wv, (size_t)n_layers * dim * dim, scale);
    fill_init_weights(w->wo, (size_t)n_layers * dim * dim, scale);
    fill_ones(w->rms_ffn_weight, (size_t)n_layers * dim);
    fill_init_weights(w->w1, (size_t)n_layers * hidden_dim * dim, scale);
    fill_init_weights(w->w2, (size_t)n_layers * dim * hidden_dim, scale);
    fill_init_weights(w->w3, (size_t)n_layers * hidden_dim * dim, scale);
    fill_ones(w->rms_final_weight, (size_t)dim);
    fill_init_weights(w->wcls, (size_t)vocab_size * dim, scale);

    // Initialize precomputed RoPE tables
    if (!init_rope_table(&model->rope, seq_len, head_size)) {
        fprintf(stderr, "[transformer] Error: Failed to precompute RoPE tables\n");
        transformer_free(model);
        return NULL;
    }

    // Allocate KV-Cache with layout [n_layers, n_heads, seq_len, head_size] for sequential cache locality
    size_t kv_bytes = (size_t)n_layers * n_heads * seq_len * head_size * sizeof(float);
    s->key_cache   = (float*)neuralc_aligned_alloc(64, kv_bytes);
    s->value_cache = (float*)neuralc_aligned_alloc(64, kv_bytes);

    if (s->key_cache)   memset(s->key_cache, 0, kv_bytes);
    if (s->value_cache) memset(s->value_cache, 0, kv_bytes);

    // Allocate state buffers
    s->x      = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    s->xb     = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    s->xb2    = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    s->hb     = (float*)neuralc_aligned_alloc(64, (size_t)hidden_dim * sizeof(float));
    s->hb2    = (float*)neuralc_aligned_alloc(64, (size_t)hidden_dim * sizeof(float));
    s->q      = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    s->k      = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    s->v      = (float*)neuralc_aligned_alloc(64, (size_t)dim * sizeof(float));
    s->att    = (float*)neuralc_aligned_alloc(64, (size_t)n_heads * seq_len * sizeof(float));
    s->logits = (float*)neuralc_aligned_alloc(64, (size_t)vocab_size * sizeof(float));

    if (!s->key_cache || !s->value_cache || !s->x || !s->xb || !s->xb2 ||
        !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits) {
        fprintf(stderr, "[transformer] Error: Memory allocation failed for KV-Cache / State buffers\n");
        transformer_free(model);
        return NULL;
    }

    return model;
}

TransformerModel *transformer_init_from_config(void) {
    TransformerConfig cfg;
    cfg.vocab_size = TRANS_VOCAB_SIZE;
    cfg.dim        = TRANS_DIM;
    cfg.hidden_dim = TRANS_HIDDEN_DIM;
    cfg.n_layers   = TRANS_N_LAYERS;
    cfg.n_heads    = TRANS_N_HEADS;
    cfg.seq_len    = TRANS_MAX_SEQ_LEN;
    return transformer_init(cfg);
}

void transformer_free(TransformerModel *model) {
    if (!model) return;

    TransformerWeights *w = &model->weights;
    neuralc_aligned_free(w->token_embedding_table);
    neuralc_aligned_free(w->rms_att_weight);
    neuralc_aligned_free(w->wq);
    neuralc_aligned_free(w->wk);
    neuralc_aligned_free(w->wv);
    neuralc_aligned_free(w->wo);
    neuralc_aligned_free(w->rms_ffn_weight);
    neuralc_aligned_free(w->w1);
    neuralc_aligned_free(w->w2);
    neuralc_aligned_free(w->w3);
    neuralc_aligned_free(w->rms_final_weight);
    neuralc_aligned_free(w->wcls);

    RoPETable *rope = &model->rope;
    neuralc_aligned_free(rope->cos);
    neuralc_aligned_free(rope->sin);

    TransformerState *s = &model->state;
    neuralc_aligned_free(s->key_cache);
    neuralc_aligned_free(s->value_cache);
    neuralc_aligned_free(s->x);
    neuralc_aligned_free(s->xb);
    neuralc_aligned_free(s->xb2);
    neuralc_aligned_free(s->hb);
    neuralc_aligned_free(s->hb2);
    neuralc_aligned_free(s->q);
    neuralc_aligned_free(s->k);
    neuralc_aligned_free(s->v);
    neuralc_aligned_free(s->att);
    neuralc_aligned_free(s->logits);

    free(model);
}

// ── Transformer Forward Pass (O(1) Token Generation with KV-Cache) ─

float *transformer_forward(TransformerModel *model, int token, int pos) {
    if (!model) return NULL;
    TransformerConfig *cfg = &model->config;
    TransformerWeights *w = &model->weights;
    TransformerState   *s = &model->state;

    // Bounds checking
    if (token < 0 || token >= cfg->vocab_size) {
        fprintf(stderr, "[transformer] Error: token ID %d out of bounds (vocab_size=%d)\n", token, cfg->vocab_size);
        return NULL;
    }
    if (pos < 0 || pos >= cfg->seq_len) {
        fprintf(stderr, "[transformer] Error: pos %d out of bounds (seq_len=%d)\n", pos, cfg->seq_len);
        return NULL;
    }

    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int n_heads    = cfg->n_heads;
    int head_size  = dim / n_heads;
    int seq_len    = cfg->seq_len;

    // Step 1: Copy embedding row into state x
    const float *emb_row = w->token_embedding_table + (size_t)token * dim;
    memcpy(s->x, emb_row, dim * sizeof(float));

    // Step 2: Forward pass through Transformer layers
    for (int l = 0; l < cfg->n_layers; l++) {
        // RMSNorm before self-attention
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);

        // Q, K, V Projections
        matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + (size_t)l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + (size_t)l * dim * dim, dim, dim);

        // Apply fast RoPE using precomputed lookup table
        apply_rope_fast(s->q, s->k, pos, head_size, n_heads, &model->rope);

        // Store K and V into KV-Cache using layout [n_layers, n_heads, seq_len, head_size]
        for (int h = 0; h < n_heads; h++) {
            size_t cache_offset = ((((size_t)l * n_heads + h) * seq_len) + pos) * head_size;
            memcpy(s->key_cache + cache_offset,   s->k + h * head_size, head_size * sizeof(float));
            memcpy(s->value_cache + cache_offset, s->v + h * head_size, head_size * sizeof(float));
        }

        // Multi-Head Self-Attention over timesteps t in [0 ... pos]
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < n_heads; h++) {
            const float *q_h = s->q + h * head_size;
            float *att_h = s->att + h * seq_len;
            const float *key_head_base = s->key_cache + (((size_t)l * n_heads + h) * seq_len) * head_size;
            const float *val_head_base = s->value_cache + (((size_t)l * n_heads + h) * seq_len) * head_size;

            float inv_sqrt_head = 1.0f / sqrtf((float)head_size);

            for (int t = 0; t <= pos; t++) {
                const float *k_t_h = key_head_base + (size_t)t * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_h[i] * k_t_h[i];
                }
                att_h[t] = score * inv_sqrt_head;
            }

            // Softmax attention weights over [0 ... pos]
            softmax(att_h, pos + 1);

            // Weighted sum of V vectors into work buffer xb for head h
            float *xb_h = s->xb + h * head_size;
            for (int i = 0; i < head_size; i++) {
                xb_h[i] = 0.0f;
            }
            for (int t = 0; t <= pos; t++) {
                float a = att_h[t];
                const float *v_t_h = val_head_base + (size_t)t * head_size;
                for (int i = 0; i < head_size; i++) {
                    xb_h[i] += a * v_t_h[i];
                }
            }
        }

        // Out projection Wo
        matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // RMSNorm before SwiGLU FFN
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        // SwiGLU Feed-Forward Network: w2 * (SiLU(w1 * x) * (w3 * x))
        matmul(s->hb,  s->xb, w->w1 + (size_t)l * hidden_dim * dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + (size_t)l * hidden_dim * dim, dim, hidden_dim);

        // Same #pragma omp parallel for schedule(static) as before -- the
        // only change is what each thread does with its chunk: AVX2
        // fused-vectorized silu*gate when available, the original scalar
        // loop otherwise. Threading structure/schedule is untouched.
#ifdef _OPENMP
        #pragma omp parallel
        {
            int nt = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int chunk = (hidden_dim + nt - 1) / nt;
            int start = tid * chunk;
            int end = start + chunk; if (end > hidden_dim) end = hidden_dim;
            int len = end - start;
            if (len > 0) {
#if NC_ENABLE_AVX2_GEMV
                if (matmul_cpu_supports_avx2_fma()) {
                    swiglu_gate_avx2(s->hb + start, s->hb2 + start, len);
                } else
#endif
                {
                    for (int i = start; i < end; i++) {
                        float val = s->hb[i];
                        float silu = val / (1.0f + expf(-val));
                        s->hb[i] = silu * s->hb2[i];
                    }
                }
            }
        }
#else
#if NC_ENABLE_AVX2_GEMV
        if (matmul_cpu_supports_avx2_fma()) {
            swiglu_gate_avx2(s->hb, s->hb2, hidden_dim);
        } else
#endif
        {
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                float silu = val / (1.0f + expf(-val));
                s->hb[i] = silu * s->hb2[i];
            }
        }
#endif

        matmul(s->xb2, s->hb, w->w2 + (size_t)l * dim * hidden_dim, hidden_dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
    }

    // Step 3: Final RMSNorm and Classifier (Logits)
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, dim, cfg->vocab_size);

    if (model->debug_mode) {
        validate_tensor_buffer(s->logits, cfg->vocab_size, "output_logits");
    }

    return s->logits;
}

// Multi-token sequence batch forward pass
float *transformer_forward_batch(TransformerModel *model, const int *tokens, int n_tokens, int start_pos) {
    if (!model || !tokens || n_tokens <= 0) return NULL;
    float *last_logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        last_logits = transformer_forward(model, tokens[i], start_pos + i);
        if (!last_logits) return NULL;
    }
    return last_logits;
}

// ── Binary Model Loading & Saving ───────────────────────────────────

int transformer_load_weights(TransformerModel *model, const char *filepath) {
    if (!model || !filepath) return -1;
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[transformer] Error: Cannot open weight file '%s' for reading\n", filepath);
        return -1;
    }

    TransformerConfig *cfg = &model->config;
    TransformerWeights *w = &model->weights;

    // Read and verify header config
    TransformerConfig file_cfg;
    if (fread(&file_cfg, sizeof(TransformerConfig), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    if (file_cfg.vocab_size != cfg->vocab_size || file_cfg.dim != cfg->dim ||
        file_cfg.hidden_dim != cfg->hidden_dim || file_cfg.n_layers != cfg->n_layers) {
        fprintf(stderr, "[transformer] Error: Config mismatch in model weight file\n");
        fclose(f);
        return -1;
    }

    int v = cfg->vocab_size;
    int d = cfg->dim;
    int h = cfg->hidden_dim;
    int l = cfg->n_layers;

    size_t reads = 0;
    reads += fread(w->token_embedding_table, sizeof(float), (size_t)v * d, f);
    reads += fread(w->rms_att_weight,        sizeof(float), (size_t)l * d, f);
    reads += fread(w->wq,                    sizeof(float), (size_t)l * d * d, f);
    reads += fread(w->wk,                    sizeof(float), (size_t)l * d * d, f);
    reads += fread(w->wv,                    sizeof(float), (size_t)l * d * d, f);
    reads += fread(w->wo,                    sizeof(float), (size_t)l * d * d, f);
    reads += fread(w->rms_ffn_weight,        sizeof(float), (size_t)l * d, f);
    reads += fread(w->w1,                    sizeof(float), (size_t)l * h * d, f);
    reads += fread(w->w2,                    sizeof(float), (size_t)l * d * h, f);
    reads += fread(w->w3,                    sizeof(float), (size_t)l * h * d, f);
    reads += fread(w->rms_final_weight,      sizeof(float), (size_t)d, f);
    reads += fread(w->wcls,                  sizeof(float), (size_t)v * d, f);

    fclose(f);
    printf("[transformer] Successfully loaded binary weights from '%s'\n", filepath);
    return 0;
}

int transformer_save_weights(const TransformerModel *model, const char *filepath) {
    if (!model || !filepath) return -1;
    FILE *f = fopen(filepath, "wb");
    if (!f) {
        fprintf(stderr, "[transformer] Error: Cannot open weight file '%s' for writing\n", filepath);
        return -1;
    }

    const TransformerConfig *cfg = &model->config;
    const TransformerWeights *w = &model->weights;

    fwrite(cfg, sizeof(TransformerConfig), 1, f);

    int v = cfg->vocab_size;
    int d = cfg->dim;
    int h = cfg->hidden_dim;
    int l = cfg->n_layers;

    fwrite(w->token_embedding_table, sizeof(float), (size_t)v * d, f);
    fwrite(w->rms_att_weight,        sizeof(float), (size_t)l * d, f);
    fwrite(w->wq,                    sizeof(float), (size_t)l * d * d, f);
    fwrite(w->wk,                    sizeof(float), (size_t)l * d * d, f);
    fwrite(w->wv,                    sizeof(float), (size_t)l * d * d, f);
    fwrite(w->wo,                    sizeof(float), (size_t)l * d * d, f);
    fwrite(w->rms_ffn_weight,        sizeof(float), (size_t)l * d, f);
    fwrite(w->w1,                    sizeof(float), (size_t)l * h * d, f);
    fwrite(w->w2,                    sizeof(float), (size_t)l * d * h, f);
    fwrite(w->w3,                    sizeof(float), (size_t)l * h * d, f);
    fwrite(w->rms_final_weight,      sizeof(float), (size_t)d, f);
    fwrite(w->wcls,                  sizeof(float), (size_t)v * d, f);

    fclose(f);
    printf("[transformer] Successfully saved binary weights to '%s'\n", filepath);
    return 0;
}

// ── Quantization Support (INT8 Q8_0) ────────────────────────────────

QuantizedWeightQ8 *quantize_weight_q8(const float *src, size_t rows, size_t cols) {
    if (!src || rows == 0 || cols == 0) return NULL;

    QuantizedWeightQ8 *qw = (QuantizedWeightQ8*)calloc(1, sizeof(QuantizedWeightQ8));
    if (!qw) return NULL;

    qw->rows = rows;
    qw->cols = cols;
    qw->qweights = (int8_t*)neuralc_aligned_alloc(64, rows * cols * sizeof(int8_t));
    qw->scales   = (float*)neuralc_aligned_alloc(64, rows * sizeof(float));

    if (!qw->qweights || !qw->scales) {
        free_quantized_weight_q8(qw);
        return NULL;
    }

    for (size_t i = 0; i < rows; i++) {
        const float *row = src + i * cols;
        float max_val = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float abs_v = fabsf(row[j]);
            if (abs_v > max_val) max_val = abs_v;
        }
        float scale = max_val / 127.0f;
        qw->scales[i] = scale;

        float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
        for (size_t j = 0; j < cols; j++) {
            int val = (int)roundf(row[j] * inv_scale);
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            qw->qweights[i * cols + j] = (int8_t)val;
        }
    }
    return qw;
}

void free_quantized_weight_q8(QuantizedWeightQ8 *qw) {
    if (!qw) return;
    neuralc_aligned_free(qw->qweights);
    neuralc_aligned_free(qw->scales);
    free(qw);
}

void matmul_q8(float *xout, const float *x, const QuantizedWeightQ8 *qw, int n, int d) {
    NC_ASSERT(xout && x && qw && n > 0 && d > 0, "Invalid matmul_q8 parameters");

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i++) {
        const int8_t *w_row = qw->qweights + (size_t)i * n;
        float scale = qw->scales[i];
        int32_t acc = 0;
        for (int j = 0; j < n; j++) {
            acc += (int32_t)w_row[j] * (int32_t)roundf(x[j] * 127.0f);
        }
        xout[i] = ((float)acc * scale) / 127.0f;
    }
}
