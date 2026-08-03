/*
 * tools/benchmark_suite.c — Comprehensive Senior C11 Performance & Architectural Benchmark Suite
 *
 * Measures:
 * 1. GEMM (Matrix Multiplication) FLOPS across thread counts (1, 2, 4, 8 OMP threads)
 * 2. Element-wise Math & Activation Latencies (Add, Mul, ReLU, Softmax)
 * 3. Decoder-Only Transformer Inference Latency (ms/token & tok/s across model sizes)
 * 4. Dense Layer Training & Forward/Backward Performance
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <omp.h>

#include "tensor.h"
#include "transformer.h"
#include "layer.h"
#include "nn.h"

// Microsecond high-resolution timer
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. GEMM Benchmark
// ─────────────────────────────────────────────────────────────────────────────
void bench_gemm(void) {
    printf("\n=========================================================\n");
    printf(" 1. GEMM (Matrix Multiplication) FLOPS & Multi-threading\n");
    printf("=========================================================\n");

    int sizes[] = {128, 256, 512, 1024};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int thread_counts[] = {1, 2, 4, 8};
    int n_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        printf("\n--- Matrix Size: %d x %d ---\n", N, N);

        Tensor *A = tensor_randn((int[]){N, N}, 2);
        Tensor *B = tensor_randn((int[]){N, N}, 2);
        Tensor *C = tensor_create((int[]){N, N}, 2);

        for (int t = 0; t < n_threads; t++) {
            int threads = thread_counts[t];
            omp_set_num_threads(threads);

            // Warmup
            tensor_matmul(A, B, C);

            int iters = (N <= 256) ? 30 : (N <= 512) ? 15 : 5;
            double t0 = get_time_sec();
            for (int i = 0; i < iters; i++) {
                tensor_matmul(A, B, C);
            }
            double t1 = get_time_sec();

            double avg_time = (t1 - t0) / iters;
            // Total floating point operations for N x N matrix multiply = 2 * N^3
            double flops = 2.0 * (double)N * (double)N * (double)N;
            double gflops = (flops / avg_time) / 1e9;

            printf(" Threads: %d | Avg Time: %7.3f ms | Throughput: %7.2f GFLOPS\n",
                   threads, avg_time * 1000.0, gflops);
        }

        tensor_free(A);
        tensor_free(B);
        tensor_free(C);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Element-wise & Activation Operations Benchmark
// ─────────────────────────────────────────────────────────────────────────────
void bench_elementwise(void) {
    printf("\n=========================================================\n");
    printf(" 2. Element-wise Math & Activation Latencies\n");
    printf("=========================================================\n");

    int dims[] = {100000, 1000000, 5000000};
    int n_dims = sizeof(dims) / sizeof(dims[0]);

    for (int d = 0; d < n_dims; d++) {
        int N = dims[d];
        printf("\n--- Element Count: %d elements (%zu MB) ---\n", N, (size_t)N * sizeof(float) / (1024 * 1024));

        Tensor *a = tensor_ones((int[]){N}, 1);
        Tensor *b = tensor_ones((int[]){N}, 1);
        Tensor *out = tensor_create((int[]){N}, 1);

        int iters = 100;
        
        // Add
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++) tensor_add(a, b, out);
        double t1 = get_time_sec();
        printf(" tensor_add:     %6.3f ms | %7.2f GB/s\n", 
               ((t1 - t0)/iters)*1000.0, (3.0 * N * sizeof(float) * iters / (t1 - t0)) / 1e9);

        // Mul
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++) tensor_mul(a, b, out);
        t1 = get_time_sec();
        printf(" tensor_mul:     %6.3f ms | %7.2f GB/s\n", 
               ((t1 - t0)/iters)*1000.0, (3.0 * N * sizeof(float) * iters / (t1 - t0)) / 1e9);

        // ReLU
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++) tensor_relu(a, out);
        t1 = get_time_sec();
        printf(" tensor_relu:    %6.3f ms | %7.2f GB/s\n", 
               ((t1 - t0)/iters)*1000.0, (2.0 * N * sizeof(float) * iters / (t1 - t0)) / 1e9);

        // Softmax (for 2D [1000, N/1000])
        if (N >= 1000) {
            Tensor *a2 = tensor_ones((int[]){1000, N / 1000}, 2);
            Tensor *out2 = tensor_create((int[]){1000, N / 1000}, 2);
            t0 = get_time_sec();
            for (int i = 0; i < iters; i++) tensor_softmax(a2, out2);
            t1 = get_time_sec();
            printf(" tensor_softmax: %6.3f ms | %7.2f Mops/s\n", 
                   ((t1 - t0)/iters)*1000.0, (N * iters / (t1 - t0)) / 1e6);
            tensor_free(a2);
            tensor_free(out2);
        }

        tensor_free(a);
        tensor_free(b);
        tensor_free(out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Decoder-Only Transformer Inference Latency
// ─────────────────────────────────────────────────────────────────────────────
void bench_transformer(void) {
    printf("\n=========================================================\n");
    printf(" 3. Decoder Transformer Engine (RoPE, RMSNorm, SwiGLU, KV-Cache)\n");
    printf("=========================================================\n");

    struct {
        const char *name;
        TransformerConfig cfg;
    } models[] = {
        {"Nano   (d=128,  L=4,  H=4)",  {1000,  128,  256,  4,  4, 512}},
        {"Micro  (d=256,  L=6,  H=8)",  {32000, 256,  512,  6,  8, 1024}},
        {"Base   (d=512,  L=8,  H=8)",  {32000, 512, 1024,  8,  8, 2048}},
        {"Large  (d=768, L=12, H=12)", {32000, 768, 2048, 12, 12, 2048}}
    };

    int num_models = sizeof(models) / sizeof(models[0]);

    for (int m = 0; m < num_models; m++) {
        printf("\n--- Model Architecture: %s ---\n", models[m].name);

        TransformerModel *model = transformer_init(models[m].cfg);
        if (!model) {
            printf(" Failed to allocate model %s\n", models[m].name);
            continue;
        }

        int n_tokens = 30;
        double t0 = get_time_sec();
        for (int pos = 0; pos < n_tokens; pos++) {
            int token = (pos * 37 + 13) % models[m].cfg.vocab_size;
            float *logits = transformer_forward(model, token, pos);
            (void)logits;
        }
        double t1 = get_time_sec();

        double total_sec = t1 - t0;
        double ms_per_token = (total_sec / n_tokens) * 1000.0;
        double tokens_per_sec = n_tokens / total_sec;

        printf(" Generation Speed: %6.2f ms/token | Throughput: %7.2f tokens/sec\n",
               ms_per_token, tokens_per_sec);

        transformer_free(model);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Dense Layer Forward & Backward Pass Performance
// ─────────────────────────────────────────────────────────────────────────────
void bench_dense_layer(void) {
    printf("\n=========================================================\n");
    printf(" 4. Dense (Linear) Layer Forward & Backward Throughput\n");
    printf("=========================================================\n");

    int batch_sizes[] = {1, 16, 64};
    int in_dims[] = {256, 512, 1024};
    int out_dims[] = {256, 512, 1024};

    for (int i = 0; i < 3; i++) {
        int B = batch_sizes[i];
        int In = in_dims[i];
        int Out = out_dims[i];

        printf("\n--- Dense Layer: Batch=%d, In=%d, Out=%d ---\n", B, In, Out);

        DenseLayer *l = dense_create(In, Out, ACT_RELU);
        dense_init_weights(l);

        Tensor *x = tensor_randn((int[]){B, In}, 2);
        Tensor *y = tensor_create((int[]){B, Out}, 2);
        Tensor *grad_out = tensor_ones((int[]){B, Out}, 2);

        int iters = 50;

        // Forward
        double t0 = get_time_sec();
        for (int k = 0; k < iters; k++) {
            dense_forward(l, x, y);
        }
        double t1 = get_time_sec();

        double fwd_ms = ((t1 - t0) / iters) * 1000.0;

        // Backward
        dense_forward(l, x, y);
        t0 = get_time_sec();
        for (int k = 0; k < iters; k++) {
            dense_backward(l, grad_out);
        }
        t1 = get_time_sec();
        double bwd_ms = ((t1 - t0) / iters) * 1000.0;

        printf(" Forward Pass:  %6.3f ms/iter\n", fwd_ms);
        printf(" Backward Pass: %6.3f ms/iter\n", bwd_ms);
        printf(" Total Pass:    %6.3f ms/iter (%7.2f passes/sec)\n", fwd_ms + bwd_ms, 1000.0 / (fwd_ms + bwd_ms));

        tensor_free(y);
        tensor_free(x);
        tensor_free(grad_out);
        dense_free(l);
    }
}

int main(void) {
    printf("=========================================================\n");
    printf("   NEURICODE / NEURALC SENIOR C11 BENCHMARK SUITE       \n");
    printf("=========================================================\n");
    printf(" OpenMP Multi-threading Enabled (Max Threads: %d)\n", omp_get_max_threads());

    bench_gemm();
    bench_elementwise();
    bench_transformer();
    bench_dense_layer();

    printf("\n=========================================================\n");
    printf("   BENCHMARK SUITE COMPLETE                             \n");
    printf("=========================================================\n");

    return 0;
}
