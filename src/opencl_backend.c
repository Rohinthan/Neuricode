/*
 * opencl_backend.c
 * ────────────────────────────────────────────────────────────────
 * OpenCL kernels + launchers backing opencl_backend.h.
 *
 * Compiled with plain gcc (OpenCL kernels are built at RUNTIME from
 * the source strings below via clBuildProgram — no special host
 * compiler needed, unlike CUDA's nvcc). Targets OpenCL 1.2 for the
 * broadest possible hardware support (old integrated GPUs, low-end
 * laptops without a discrete NVIDIA card, etc.) — this is the whole
 * point of having this backend alongside CUDA.
 *
 * Build: gcc -c opencl_backend.c -o opencl_backend.o
 * Link:  -lOpenCL
 */

#define CL_TARGET_OPENCL_VERSION 120
#include "opencl_backend.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── error handling — mirrors cuda_backend.c's fail-fast style ───── */
#define CL_CHECK(err, what)                                                \
    do {                                                                   \
        if ((err) != CL_SUCCESS) {                                        \
            fprintf(stderr, "[OpenCL] %s:%d: %s failed (err %d)\n",       \
                    __FILE__, __LINE__, (what), (int)(err));              \
            exit(1);                                                      \
        }                                                                  \
    } while (0)

/* ── lazily-initialized global context ──────────────────────────── */
static cl_platform_id   g_platform   = NULL;
static cl_device_id     g_device     = NULL;
static cl_context       g_context    = NULL;
static cl_command_queue g_queue      = NULL;
static cl_program       g_program    = NULL;
static int              g_inited     = 0;   /* 1 once init succeeded */
static int              g_init_tried = 0;   /* 1 once init was attempted */
static char             g_device_name[256] = "none";

/* ── kernel source (OpenCL C, built once at first use) ───────────── */
static const char *KERNEL_SRC =
"__kernel void k_matmul(__global const float *A, __global const float *B,\n"
"                        __global float *C, int M, int K, int N) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row >= M || col >= N) return;\n"
"    float acc = 0.0f;\n"
"    for (int k = 0; k < K; k++)\n"
"        acc += A[row*K + k] * B[k*N + col];\n"
"    C[row*N + col] = acc;\n"
"}\n"
"\n"
"__kernel void k_linear_forward(__global const float *x, __global const float *W,\n"
"                                __global const float *bias, __global float *z,\n"
"                                int batch, int in_f, int out_f) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= batch * out_f) return;\n"
"    int b = idx / out_f, o = idx % out_f;\n"
"    float acc = bias[o];\n"
"    __global const float *xr = x + (size_t)b * in_f;\n"
"    __global const float *wr = W + (size_t)o * in_f;\n"
"    for (int i = 0; i < in_f; i++) acc += xr[i] * wr[i];\n"
"    z[idx] = acc;\n"
"}\n"
"\n"
"__kernel void k_linear_dW(__global const float *x, __global const float *dz,\n"
"                           __global float *dW, int batch, int in_f, int out_f) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= out_f * in_f) return;\n"
"    int o = idx / in_f, i = idx % in_f;\n"
"    float acc = 0.0f;\n"
"    for (int b = 0; b < batch; b++)\n"
"        acc += dz[(size_t)b*out_f + o] * x[(size_t)b*in_f + i];\n"
"    dW[idx] = acc / (float)batch;\n"
"}\n"
"\n"
"__kernel void k_linear_db(__global const float *dz, __global float *db,\n"
"                           int batch, int out_f) {\n"
"    int o = get_global_id(0);\n"
"    if (o >= out_f) return;\n"
"    float acc = 0.0f;\n"
"    for (int b = 0; b < batch; b++) acc += dz[(size_t)b*out_f + o];\n"
"    db[o] = acc / (float)batch;\n"
"}\n"
"\n"
"__kernel void k_linear_dX(__global const float *W, __global const float *dz,\n"
"                           __global float *dX, int batch, int in_f, int out_f) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx >= batch * in_f) return;\n"
"    int b = idx / in_f, i = idx % in_f;\n"
"    float acc = 0.0f;\n"
"    for (int o = 0; o < out_f; o++)\n"
"        acc += dz[(size_t)b*out_f + o] * W[(size_t)o*in_f + i];\n"
"    dX[idx] = acc;\n"
"}\n"
"\n"
"__kernel void k_add(__global const float *a, __global const float *b,\n"
"                     __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] + b[i];\n"
"}\n"
"__kernel void k_sub(__global const float *a, __global const float *b,\n"
"                     __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] - b[i];\n"
"}\n"
"__kernel void k_mul(__global const float *a, __global const float *b,\n"
"                     __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] * b[i];\n"
"}\n"
"__kernel void k_scale(__global const float *a, float s, __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] * s;\n"
"}\n"
"__kernel void k_add_scalar(__global const float *a, float s, __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] + s;\n"
"}\n"
"__kernel void k_fill(__global float *a, float val, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) a[i] = val;\n"
"}\n"
"\n"
"__kernel void k_relu(__global const float *a, __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] > 0.0f ? a[i] : 0.0f;\n"
"}\n"
"__kernel void k_relu_grad(__global const float *a, __global const float *grad,\n"
"                           __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = a[i] > 0.0f ? grad[i] : 0.0f;\n"
"}\n"
"__kernel void k_sigmoid(__global const float *a, __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = 1.0f / (1.0f + exp(-a[i]));\n"
"}\n"
"__kernel void k_sigmoid_grad(__global const float *sig, __global const float *grad,\n"
"                              __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0);\n"
"    if (i < n) out[i] = grad[i] * sig[i] * (1.0f - sig[i]);\n"
"}\n"
"__kernel void k_tanh(__global const float *a, __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0); if (i < n) out[i] = tanh(a[i]);\n"
"}\n"
"__kernel void k_tanh_grad(__global const float *th, __global const float *grad,\n"
"                           __global float *out, size_t n) {\n"
"    size_t i = get_global_id(0);\n"
"    if (i < n) out[i] = grad[i] * (1.0f - th[i]*th[i]);\n"
"}\n"
"\n"
"__kernel void k_softmax_rows(__global const float *a, __global float *out,\n"
"                              int cols, __local float *sh) {\n"
"    int row = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    int lsz = get_local_size(0);\n"
"    __global const float *in_row  = a   + (size_t)row * cols;\n"
"    __global float       *out_row = out + (size_t)row * cols;\n"
"\n"
"    float local_max = -INFINITY;\n"
"    for (int j = tid; j < cols; j += lsz)\n"
"        local_max = fmax(local_max, in_row[j]);\n"
"    sh[tid] = local_max;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = lsz/2; s > 0; s >>= 1) {\n"
"        if (tid < s) sh[tid] = fmax(sh[tid], sh[tid+s]);\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    float row_max = sh[0];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    float local_sum = 0.0f;\n"
"    for (int j = tid; j < cols; j += lsz) {\n"
"        float e = exp(in_row[j] - row_max);\n"
"        out_row[j] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    sh[tid] = local_sum;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int s = lsz/2; s > 0; s >>= 1) {\n"
"        if (tid < s) sh[tid] += sh[tid+s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    float row_sum = sh[0];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    for (int j = tid; j < cols; j += lsz)\n"
"        out_row[j] /= row_sum;\n"
"}\n";

/* ── lazy init ──────────────────────────────────────────────────── */

static int cl_lazy_init(void) {
    if (g_init_tried) return g_inited;
    g_init_tried = 1;

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &n_platforms) != CL_SUCCESS || n_platforms == 0)
        return 0;
    cl_platform_id *platforms = malloc(n_platforms * sizeof(cl_platform_id));
    if (!platforms) return 0;
    clGetPlatformIDs(n_platforms, platforms, NULL);

    /* pick the first platform that has ANY device (GPU preferred,
     * fall back to CPU device via the OpenCL ICD — still faster than
     * nothing on some low-end boxes with only a CPU OpenCL runtime) */
    for (cl_uint p = 0; p < n_platforms && !g_device; p++) {
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &g_device, NULL) == CL_SUCCESS)
            g_platform = platforms[p];
        else if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 1, &g_device, NULL) == CL_SUCCESS)
            g_platform = platforms[p];
    }
    free(platforms);
    if (!g_device) return 0;

    clGetDeviceInfo(g_device, CL_DEVICE_NAME, sizeof(g_device_name), g_device_name, NULL);

    cl_int err;
    g_context = clCreateContext(NULL, 1, &g_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;

    g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
    if (err != CL_SUCCESS) return 0;

    g_program = clCreateProgramWithSource(g_context, 1, &KERNEL_SRC, NULL, &err);
    if (err != CL_SUCCESS) return 0;

    err = clBuildProgram(g_program, 1, &g_device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG,
                               sizeof(log), log, NULL);
        fprintf(stderr, "[OpenCL] kernel build failed:\n%s\n", log);
        return 0;
    }

    g_inited = 1;
    return 1;
}

static cl_kernel get_kernel(const char *name) {
    cl_int err;
    cl_kernel k = clCreateKernel(g_program, name, &err);
    CL_CHECK(err, name);
    return k;
}

/* ══════════════════════════════════════════════════════════════════
 * device / memory management
 * ════════════════════════════════════════════════════════════════ */

int opencl_available(void) { return cl_lazy_init(); }

const char *opencl_device_name(void) {
    cl_lazy_init();
    return g_device_name;
}

float *opencl_malloc_f32(size_t n) {
    if (!cl_lazy_init()) {
        fprintf(stderr, "[OpenCL] no device available\n");
        exit(1);
    }
    cl_int err;
    cl_mem buf = clCreateBuffer(g_context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, &err);
    CL_CHECK(err, "clCreateBuffer");
    return (float *)buf;
}

void opencl_free_f32(float *dev_buf) {
    if (dev_buf) clReleaseMemObject((cl_mem)dev_buf);
}

void opencl_memcpy_h2d(float *dst_dev, const float *src_host, size_t n) {
    cl_int err = clEnqueueWriteBuffer(g_queue, (cl_mem)dst_dev, CL_TRUE, 0,
                                       n * sizeof(float), src_host, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueWriteBuffer");
}

void opencl_memcpy_d2h(float *dst_host, const float *src_dev, size_t n) {
    cl_int err = clEnqueueReadBuffer(g_queue, (cl_mem)src_dev, CL_TRUE, 0,
                                      n * sizeof(float), dst_host, 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueReadBuffer");
}

void opencl_memcpy_d2d(float *dst_dev, const float *src_dev, size_t n) {
    cl_int err = clEnqueueCopyBuffer(g_queue, (cl_mem)src_dev, (cl_mem)dst_dev,
                                      0, 0, n * sizeof(float), 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueCopyBuffer");
}

void opencl_memset_zero(float *dev_buf, size_t n) {
    float zero = 0.0f;
    cl_int err = clEnqueueFillBuffer(g_queue, (cl_mem)dev_buf, &zero, sizeof(float),
                                      0, n * sizeof(float), 0, NULL, NULL);
    CL_CHECK(err, "clEnqueueFillBuffer");
}

void opencl_sync(void) { clFinish(g_queue); }

/* ══════════════════════════════════════════════════════════════════
 * linear algebra
 * ════════════════════════════════════════════════════════════════ */

void opencl_matmul(const float *a, const float *b, float *out, int M, int K, int N) {
    cl_kernel k = get_kernel("k_matmul");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b);
    clSetKernelArg(k, 2, sizeof(cl_mem), &out);
    clSetKernelArg(k, 3, sizeof(int), &M);
    clSetKernelArg(k, 4, sizeof(int), &K);
    clSetKernelArg(k, 5, sizeof(int), &N);
    size_t global[2] = { (size_t)M, (size_t)N };
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 2, NULL, global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_matmul");
    clReleaseKernel(k);
}

/* ══════════════════════════════════════════════════════════════════
 * dense (fully-connected) layer
 * ════════════════════════════════════════════════════════════════ */

void opencl_linear_forward(const float *x, const float *W, const float *bias,
                            float *z, int batch, int in_f, int out_f) {
    cl_kernel k = get_kernel("k_linear_forward");
    clSetKernelArg(k, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k, 1, sizeof(cl_mem), &W);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bias);
    clSetKernelArg(k, 3, sizeof(cl_mem), &z);
    clSetKernelArg(k, 4, sizeof(int), &batch);
    clSetKernelArg(k, 5, sizeof(int), &in_f);
    clSetKernelArg(k, 6, sizeof(int), &out_f);
    size_t global = (size_t)batch * out_f;
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_linear_forward");
    clReleaseKernel(k);
}

void opencl_linear_backward(const float *x, const float *W, const float *dz,
                             float *dW, float *db, float *dX,
                             int batch, int in_f, int out_f) {
    cl_int err;

    cl_kernel k1 = get_kernel("k_linear_dW");
    clSetKernelArg(k1, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k1, 1, sizeof(cl_mem), &dz);
    clSetKernelArg(k1, 2, sizeof(cl_mem), &dW);
    clSetKernelArg(k1, 3, sizeof(int), &batch);
    clSetKernelArg(k1, 4, sizeof(int), &in_f);
    clSetKernelArg(k1, 5, sizeof(int), &out_f);
    size_t g1 = (size_t)out_f * in_f;
    err = clEnqueueNDRangeKernel(g_queue, k1, 1, NULL, &g1, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_linear_dW");
    clReleaseKernel(k1);

    cl_kernel k2 = get_kernel("k_linear_db");
    clSetKernelArg(k2, 0, sizeof(cl_mem), &dz);
    clSetKernelArg(k2, 1, sizeof(cl_mem), &db);
    clSetKernelArg(k2, 2, sizeof(int), &batch);
    clSetKernelArg(k2, 3, sizeof(int), &out_f);
    size_t g2 = (size_t)out_f;
    err = clEnqueueNDRangeKernel(g_queue, k2, 1, NULL, &g2, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_linear_db");
    clReleaseKernel(k2);

    cl_kernel k3 = get_kernel("k_linear_dX");
    clSetKernelArg(k3, 0, sizeof(cl_mem), &W);
    clSetKernelArg(k3, 1, sizeof(cl_mem), &dz);
    clSetKernelArg(k3, 2, sizeof(cl_mem), &dX);
    clSetKernelArg(k3, 3, sizeof(int), &batch);
    clSetKernelArg(k3, 4, sizeof(int), &in_f);
    clSetKernelArg(k3, 5, sizeof(int), &out_f);
    size_t g3 = (size_t)batch * in_f;
    err = clEnqueueNDRangeKernel(g_queue, k3, 1, NULL, &g3, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_linear_dX");
    clReleaseKernel(k3);
}

/* ══════════════════════════════════════════════════════════════════
 * elementwise ops
 * ════════════════════════════════════════════════════════════════ */

#define ELEMWISE_BINARY(NAME, KNAME)                                       \
    void opencl_##NAME(const float *a, const float *b, float *out, size_t n) { \
        cl_kernel k = get_kernel(KNAME);                                    \
        clSetKernelArg(k, 0, sizeof(cl_mem), &a);                           \
        clSetKernelArg(k, 1, sizeof(cl_mem), &b);                           \
        clSetKernelArg(k, 2, sizeof(cl_mem), &out);                         \
        clSetKernelArg(k, 3, sizeof(size_t), &n);                           \
        cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL); \
        CL_CHECK(err, KNAME);                                               \
        clReleaseKernel(k);                                                 \
    }

ELEMWISE_BINARY(add, "k_add")
ELEMWISE_BINARY(sub, "k_sub")
ELEMWISE_BINARY(mul, "k_mul")

void opencl_scale(const float *a, float s, float *out, size_t n) {
    cl_kernel k = get_kernel("k_scale");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(float), &s);
    clSetKernelArg(k, 2, sizeof(cl_mem), &out);
    clSetKernelArg(k, 3, sizeof(size_t), &n);
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_scale");
    clReleaseKernel(k);
}

void opencl_add_scalar(const float *a, float s, float *out, size_t n) {
    cl_kernel k = get_kernel("k_add_scalar");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(float), &s);
    clSetKernelArg(k, 2, sizeof(cl_mem), &out);
    clSetKernelArg(k, 3, sizeof(size_t), &n);
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_add_scalar");
    clReleaseKernel(k);
}

void opencl_fill(float *a, float val, size_t n) {
    cl_kernel k = get_kernel("k_fill");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(float), &val);
    clSetKernelArg(k, 2, sizeof(size_t), &n);
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_fill");
    clReleaseKernel(k);
}

/* ══════════════════════════════════════════════════════════════════
 * activations
 * ════════════════════════════════════════════════════════════════ */

#define ELEMWISE_UNARY(NAME, KNAME)                                        \
    void opencl_##NAME(const float *a, float *out, size_t n) {             \
        cl_kernel k = get_kernel(KNAME);                                    \
        clSetKernelArg(k, 0, sizeof(cl_mem), &a);                           \
        clSetKernelArg(k, 1, sizeof(cl_mem), &out);                         \
        clSetKernelArg(k, 2, sizeof(size_t), &n);                           \
        cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL); \
        CL_CHECK(err, KNAME);                                               \
        clReleaseKernel(k);                                                 \
    }

ELEMWISE_UNARY(relu, "k_relu")
ELEMWISE_UNARY(sigmoid, "k_sigmoid")

void opencl_tanh_f(const float *a, float *out, size_t n) {
    cl_kernel k = get_kernel("k_tanh");
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &out);
    clSetKernelArg(k, 2, sizeof(size_t), &n);
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL);
    CL_CHECK(err, "k_tanh");
    clReleaseKernel(k);
}

#define ELEMWISE_GRAD(NAME, KNAME)                                         \
    void opencl_##NAME(const float *a, const float *grad, float *out, size_t n) { \
        cl_kernel k = get_kernel(KNAME);                                    \
        clSetKernelArg(k, 0, sizeof(cl_mem), &a);                           \
        clSetKernelArg(k, 1, sizeof(cl_mem), &grad);                        \
        clSetKernelArg(k, 2, sizeof(cl_mem), &out);                         \
        clSetKernelArg(k, 3, sizeof(size_t), &n);                           \
        cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &n, NULL, 0, NULL, NULL); \
        CL_CHECK(err, KNAME);                                               \
        clReleaseKernel(k);                                                 \
    }

ELEMWISE_GRAD(relu_grad, "k_relu_grad")
ELEMWISE_GRAD(sigmoid_grad, "k_sigmoid_grad")
ELEMWISE_GRAD(tanh_grad, "k_tanh_grad")

void opencl_softmax_rows(const float *a, float *out, int rows, int cols) {
    cl_kernel k = get_kernel("k_softmax_rows");
    int local_size = 128;
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &out);
    clSetKernelArg(k, 2, sizeof(int), &cols);
    clSetKernelArg(k, 3, (size_t)local_size * sizeof(float), NULL); /* __local sh[] */
    size_t global = (size_t)rows * local_size;
    size_t local  = (size_t)local_size;
    cl_int err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &global, &local, 0, NULL, NULL);
    CL_CHECK(err, "k_softmax_rows");
    clReleaseKernel(k);
}
