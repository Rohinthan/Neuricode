#ifdef USE_OPENCL

#include "neuralc_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── error string ────────────────────────────────────────────────── */

const char *gpu_cl_error_string(int err) {
    switch (err) {
    case  0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -13: return "CL_INVALID_BINARY";
    case -44: return "CL_INVALID_KERNEL_NAME";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    default:  return "UNKNOWN_CL_ERROR";
    }
}

#define CL_CHECK(err, msg)                                          \
    do { if ((err) != CL_SUCCESS) {                                 \
        fprintf(stderr, "[neuralc GPU] %s: %s (code %d)\n",        \
                (msg), gpu_cl_error_string(err), (err));            \
        exit(1);                                                    \
    }} while(0)

/* ── load kernel source from file ───────────────────────────────── */

static char *load_kernel_source(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[neuralc GPU] Cannot open kernel file: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *src = (char *)malloc(sz + 1);
    if (!src) { fclose(f); return NULL; }
    if (fread(src, 1, sz, f) != (size_t)sz) {
        free(src); fclose(f); return NULL;
    }
    src[sz] = '\0';
    fclose(f);
    return src;
}

/* ── gpu_init ────────────────────────────────────────────────────── */

int gpu_init(GPUContext *ctx, const char *kernel_path) {
    cl_int err;
    memset(ctx, 0, sizeof(GPUContext));

    /* platform */
    err = clGetPlatformIDs(1, &ctx->platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[neuralc GPU] No OpenCL platform found: %s\n",
                gpu_cl_error_string(err));
        return -1;
    }

    /* device — prefer GPU, fall back to CPU */
    err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1,
                         &ctx->device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[neuralc GPU] No GPU found, trying CPU...\n");
        err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_CPU, 1,
                             &ctx->device, NULL);
        CL_CHECK(err, "clGetDeviceIDs (CPU fallback)");
    }

    /* context + queue */
    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");
    ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    /* load and compile kernels */
    char *src = load_kernel_source(kernel_path);
    if (!src) {
        fprintf(stderr, "[neuralc GPU] Failed to load %s\n", kernel_path);
        return -1;
    }
    ctx->program = clCreateProgramWithSource(ctx->context, 1,
                       (const char **)&src, NULL, &err);
    free(src);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(ctx->program, 1, &ctx->device,
                         "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        /* print build log */
        size_t log_sz;
        clGetProgramBuildInfo(ctx->program, ctx->device,
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = (char *)malloc(log_sz + 1);
        clGetProgramBuildInfo(ctx->program, ctx->device,
                              CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        log[log_sz] = '\0';
        fprintf(stderr, "[neuralc GPU] Build error:\n%s\n", log);
        free(log);
        return -1;
    }

    /* create kernel objects */
    #define MK(field, name) \
        ctx->field = clCreateKernel(ctx->program, name, &err); \
        CL_CHECK(err, "clCreateKernel:" name)

    MK(k_add,          "tensor_add_gpu");
    MK(k_sub,          "tensor_sub_gpu");
    MK(k_mul,          "tensor_mul_gpu");
    MK(k_scale,        "tensor_scale_gpu");
    MK(k_relu,         "relu_gpu");
    MK(k_relu_grad,    "relu_grad_gpu");
    MK(k_sigmoid,      "sigmoid_gpu");
    MK(k_sigmoid_grad, "sigmoid_grad_gpu");
    MK(k_tanh,         "tanh_gpu");
    MK(k_matmul,       "matmul_gpu");
    MK(k_matmul_tiled, "matmul_tiled_gpu");
    MK(k_dense_fwd,    "dense_forward_gpu");
    MK(k_softmax,      "softmax_gpu");
    #undef MK

    gpu_print_device_info(ctx);
    return 0;
}

/* ── gpu_shutdown ────────────────────────────────────────────────── */

void gpu_shutdown(GPUContext *ctx) {
    #define REL_K(k) if (ctx->k) clReleaseKernel(ctx->k)
    REL_K(k_add); REL_K(k_sub); REL_K(k_mul); REL_K(k_scale);
    REL_K(k_relu); REL_K(k_relu_grad);
    REL_K(k_sigmoid); REL_K(k_sigmoid_grad); REL_K(k_tanh);
    REL_K(k_matmul); REL_K(k_matmul_tiled);
    REL_K(k_dense_fwd); REL_K(k_softmax);
    #undef REL_K
    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue)   clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);
    memset(ctx, 0, sizeof(GPUContext));
}

void gpu_print_device_info(const GPUContext *ctx) {
    char name[256];
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    cl_ulong mem;
    clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(mem), &mem, NULL);
    printf("[neuralc GPU] Device: %s  (%.0f MB)\n",
           name, (double)mem / (1024*1024));
}

/* ── memory ──────────────────────────────────────────────────────── */

GPUTensor *gpu_tensor_alloc(GPUContext *ctx,
                             const int *shape, int ndim) {
    GPUTensor *t = (GPUTensor *)calloc(1, sizeof(GPUTensor));
    t->ndim = ndim;
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size    *= shape[i];
    }
    cl_int err;
    t->buf = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE,
                            t->size * sizeof(float), NULL, &err);
    CL_CHECK(err, "gpu_tensor_alloc: clCreateBuffer");
    return t;
}

GPUTensor *gpu_tensor_upload(GPUContext *ctx, const Tensor *cpu_t) {
    GPUTensor *t = gpu_tensor_alloc(ctx, cpu_t->shape, cpu_t->ndim);
    cl_int err = clEnqueueWriteBuffer(ctx->queue, t->buf, CL_TRUE,
                     0, cpu_t->size * sizeof(float),
                     cpu_t->data, 0, NULL, NULL);
    CL_CHECK(err, "gpu_tensor_upload");
    return t;
}

void gpu_tensor_download(GPUContext *ctx, const GPUTensor *gpu_t,
                         Tensor *cpu_out) {
    cl_int err = clEnqueueReadBuffer(ctx->queue, gpu_t->buf, CL_TRUE,
                     0, gpu_t->size * sizeof(float),
                     cpu_out->data, 0, NULL, NULL);
    CL_CHECK(err, "gpu_tensor_download");
}

void gpu_tensor_free(GPUTensor *t) {
    if (!t) return;
    if (t->buf) clReleaseMemObject(t->buf);
    free(t);
}

/* ── sync ────────────────────────────────────────────────────────── */

void gpu_sync(GPUContext *ctx) {
    clFinish(ctx->queue);
}

/* ── element-wise helpers ────────────────────────────────────────── */

static void run_elwise2(GPUContext *ctx, cl_kernel k,
                        const GPUTensor *a, const GPUTensor *b,
                        GPUTensor *out) {
    cl_int n   = (cl_int)a->size;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a->buf);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &b->buf);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &out->buf);
    err |= clSetKernelArg(k, 3, sizeof(cl_int), &n);
    CL_CHECK(err, "run_elwise2: clSetKernelArg");
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(ctx->queue, k, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "run_elwise2: clEnqueueNDRangeKernel");
}

static void run_elwise1(GPUContext *ctx, cl_kernel k,
                        const GPUTensor *a, GPUTensor *out) {
    cl_int n   = (cl_int)a->size;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a->buf);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &out->buf);
    err |= clSetKernelArg(k, 2, sizeof(cl_int), &n);
    CL_CHECK(err, "run_elwise1: clSetKernelArg");
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(ctx->queue, k, 1, NULL,
                                 &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "run_elwise1: clEnqueueNDRangeKernel");
}

/* ── GPU ops ─────────────────────────────────────────────────────── */

void gpu_add(GPUContext *ctx,
             const GPUTensor *a, const GPUTensor *b, GPUTensor *out)
{ run_elwise2(ctx, ctx->k_add, a, b, out); }

void gpu_sub(GPUContext *ctx,
             const GPUTensor *a, const GPUTensor *b, GPUTensor *out)
{ run_elwise2(ctx, ctx->k_sub, a, b, out); }

void gpu_mul(GPUContext *ctx,
             const GPUTensor *a, const GPUTensor *b, GPUTensor *out)
{ run_elwise2(ctx, ctx->k_mul, a, b, out); }

void gpu_scale(GPUContext *ctx,
               const GPUTensor *a, float s, GPUTensor *out) {
    cl_int n   = (cl_int)a->size;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_scale, 0, sizeof(cl_mem),   &a->buf);
    err |= clSetKernelArg(ctx->k_scale, 1, sizeof(cl_float), &s);
    err |= clSetKernelArg(ctx->k_scale, 2, sizeof(cl_mem),   &out->buf);
    err |= clSetKernelArg(ctx->k_scale, 3, sizeof(cl_int),   &n);
    CL_CHECK(err, "gpu_scale: clSetKernelArg");
    size_t global = (size_t)n;
    clEnqueueNDRangeKernel(ctx->queue, ctx->k_scale, 1,
                           NULL, &global, NULL, 0, NULL, NULL);
}

void gpu_relu(GPUContext *ctx,
              const GPUTensor *in, GPUTensor *out)
{ run_elwise1(ctx, ctx->k_relu, in, out); }

void gpu_relu_grad(GPUContext *ctx,
                   const GPUTensor *in, const GPUTensor *grad,
                   GPUTensor *out)
{ run_elwise2(ctx, ctx->k_relu_grad, in, grad, out); }

void gpu_sigmoid(GPUContext *ctx,
                 const GPUTensor *in, GPUTensor *out)
{ run_elwise1(ctx, ctx->k_sigmoid, in, out); }

void gpu_sigmoid_grad(GPUContext *ctx,
                      const GPUTensor *sig, const GPUTensor *grad,
                      GPUTensor *out)
{ run_elwise2(ctx, ctx->k_sigmoid_grad, sig, grad, out); }

void gpu_tanh(GPUContext *ctx,
              const GPUTensor *in, GPUTensor *out)
{ run_elwise1(ctx, ctx->k_tanh, in, out); }

/* ── matmul ──────────────────────────────────────────────────────── */

void gpu_matmul(GPUContext *ctx,
                const GPUTensor *a, const GPUTensor *b,
                GPUTensor *out, int M, int K, int N) {
    cl_int cm = M, ck = K, cn = N;
    /* use tiled kernel for large matrices */
    int use_tiled = (M >= 32 && N >= 32 && K >= 32);
    cl_kernel k   = use_tiled ? ctx->k_matmul_tiled : ctx->k_matmul;

    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(k, 0, sizeof(cl_mem), &a->buf);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &b->buf);
    err |= clSetKernelArg(k, 2, sizeof(cl_mem), &out->buf);
    err |= clSetKernelArg(k, 3, sizeof(cl_int), &cm);
    err |= clSetKernelArg(k, 4, sizeof(cl_int), &ck);
    err |= clSetKernelArg(k, 5, sizeof(cl_int), &cn);
    CL_CHECK(err, "gpu_matmul: clSetKernelArg");

    size_t tile = 16;
    size_t gws[2] = {
        ((size_t)M + tile-1) / tile * tile,
        ((size_t)N + tile-1) / tile * tile
    };
    size_t lws[2] = {tile, tile};
    err = clEnqueueNDRangeKernel(ctx->queue, k, 2, NULL,
                                 gws, use_tiled ? lws : NULL,
                                 0, NULL, NULL);
    CL_CHECK(err, "gpu_matmul: clEnqueueNDRangeKernel");
}

/* ── dense forward ───────────────────────────────────────────────── */

void gpu_dense_forward(GPUContext *ctx,
                       const GPUTensor *input,
                       const GPUTensor *W,
                       const GPUTensor *bias,
                       GPUTensor *output,
                       int batch, int in_f, int out_f) {
    cl_int cb = batch, ci = in_f, co = out_f;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_dense_fwd, 0, sizeof(cl_mem), &input->buf);
    err |= clSetKernelArg(ctx->k_dense_fwd, 1, sizeof(cl_mem), &W->buf);
    err |= clSetKernelArg(ctx->k_dense_fwd, 2, sizeof(cl_mem), &bias->buf);
    err |= clSetKernelArg(ctx->k_dense_fwd, 3, sizeof(cl_mem), &output->buf);
    err |= clSetKernelArg(ctx->k_dense_fwd, 4, sizeof(cl_int), &cb);
    err |= clSetKernelArg(ctx->k_dense_fwd, 5, sizeof(cl_int), &ci);
    err |= clSetKernelArg(ctx->k_dense_fwd, 6, sizeof(cl_int), &co);
    CL_CHECK(err, "gpu_dense_forward: clSetKernelArg");

    size_t gws[2] = {(size_t)batch, (size_t)out_f};
    err = clEnqueueNDRangeKernel(ctx->queue, ctx->k_dense_fwd,
                                 2, NULL, gws, NULL, 0, NULL, NULL);
    CL_CHECK(err, "gpu_dense_forward: clEnqueueNDRangeKernel");
}

/* ── softmax ─────────────────────────────────────────────────────── */

void gpu_softmax(GPUContext *ctx,
                 const GPUTensor *in, GPUTensor *out,
                 int rows, int cols) {
    cl_int cr = rows, cc = cols;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(ctx->k_softmax, 0, sizeof(cl_mem), &in->buf);
    err |= clSetKernelArg(ctx->k_softmax, 1, sizeof(cl_mem), &out->buf);
    err |= clSetKernelArg(ctx->k_softmax, 2, sizeof(cl_int), &cr);
    err |= clSetKernelArg(ctx->k_softmax, 3, sizeof(cl_int), &cc);
    CL_CHECK(err, "gpu_softmax: clSetKernelArg");
    size_t global = (size_t)rows;
    err = clEnqueueNDRangeKernel(ctx->queue, ctx->k_softmax,
                                 1, NULL, &global, NULL, 0, NULL, NULL);
    CL_CHECK(err, "gpu_softmax: clEnqueueNDRangeKernel");
}

#endif /* USE_OPENCL */
