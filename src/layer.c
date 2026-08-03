#include "layer.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "cuda_backend.h"
#endif

#ifdef USE_OPENCL
#include "opencl_backend.h"
#endif

/* ── helpers ─────────────────────────────────────────────────────── */

static Tensor *alloc2(int r, int c) {
    int sh[2] = {r, c};
    return tensor_zeros(sh, 2);
}
static Tensor *alloc1(int n) {
    int sh[1] = {n};
    return tensor_zeros(sh, 1);
}

/* alloc2, but resident on the given device/backend (GPU alloc skips
 * the host zero-fill tensor_zeros() would otherwise do) */
static Tensor *alloc2_dev(int r, int c, CF_Device device, CF_GpuBackend backend) {
    Tensor *t = alloc2(r, c);
    if (device == CF_DEVICE_GPU) tensor_to_gpu_ex(t, backend);
    return t;
}

/* ── lifecycle ───────────────────────────────────────────────────── */

DenseLayer *dense_create(int in_features, int out_features, Activation act) {
    DenseLayer *l = (DenseLayer *)calloc(1, sizeof(DenseLayer));
    if (!l) return NULL;
    l->in_features  = in_features;
    l->out_features = out_features;
    l->activation   = act;

    l->W  = alloc2(out_features, in_features);
    l->b  = alloc1(out_features);
    l->dW = alloc2(out_features, in_features);
    l->db = alloc1(out_features);

    dense_init_weights(l);
    return l;
}

void dense_free(DenseLayer *l) {
    if (!l) return;
    tensor_free(l->W);
    tensor_free(l->b);
    tensor_free(l->dW);
    tensor_free(l->db);
    tensor_free(l->z);
    tensor_free(l->a);
    tensor_free(l->dX);
    free(l);
}

void dense_init_weights(DenseLayer *l) {
    float scale;
    if (l->activation == ACT_RELU)
        scale = sqrtf(2.0f / l->in_features);          /* He   */
    else
        scale = sqrtf(1.0f / l->in_features);           /* Xavier */

    /* use tensor_randn then scale */
    int sh[2] = {l->out_features, l->in_features};
    Tensor *tmp = tensor_randn(sh, 2);
    tensor_scale(tmp, scale, l->W);
    tensor_free(tmp);

    tensor_fill(l->b, 0.0f);
}

/* ── GPU residency ───────────────────────────────────────────────── */

void dense_to_gpu(DenseLayer *l) {
#if !defined(USE_CUDA) && !defined(USE_OPENCL)
    (void)l;
    cforge_error("dense_to_gpu: library was built without -DUSE_CUDA or "
                 "-DUSE_OPENCL", __FILE__, __LINE__);
#else
    tensor_to_gpu(l->W);
    tensor_to_gpu(l->b);
    tensor_to_gpu(l->dW);
    tensor_to_gpu(l->db);
    /* z/a/dX are (re)allocated lazily by forward/backward on the
     * device/backend that matches the input, so nothing to do here. */
#endif
}

void dense_to_gpu_ex(DenseLayer *l, CF_GpuBackend backend) {
    tensor_to_gpu_ex(l->W, backend);
    tensor_to_gpu_ex(l->b, backend);
    tensor_to_gpu_ex(l->dW, backend);
    tensor_to_gpu_ex(l->db, backend);
}

void dense_to_cpu(DenseLayer *l) {
#if !defined(USE_CUDA) && !defined(USE_OPENCL)
    (void)l;
    cforge_error("dense_to_cpu: library was built without -DUSE_CUDA or "
                 "-DUSE_OPENCL", __FILE__, __LINE__);
#else
    tensor_to_cpu(l->W);
    tensor_to_cpu(l->b);
    tensor_to_cpu(l->dW);
    tensor_to_cpu(l->db);
    if (l->z) tensor_to_cpu(l->z);
    if (l->a) tensor_to_cpu(l->a);
    if (l->dX) tensor_to_cpu(l->dX);
#endif
}

/* ── forward pass ────────────────────────────────────────────────── */
/*
 *  z = input @ W^T + b          [batch, out]
 *  a = activation(z)
 */
void dense_forward(DenseLayer *l, const Tensor *input, Tensor *output) {
    int batch = input->shape[0];
    int in_f  = input->shape[1];
    int out_f = l->out_features;
    assert(in_f == l->in_features);
    assert(output->shape[0] == batch && output->shape[1] == out_f);

    /* cache input pointer (not owned) */
    l->input = (Tensor *)input;   /* safe: we won't free it */

    /* allocate / resize z and a if needed, on the same device+backend as input */
    if (!l->z || l->z->shape[0] != batch || l->z->device != input->device ||
        l->z->gpu_backend != input->gpu_backend) {
        tensor_free(l->z); tensor_free(l->a);
        l->z = alloc2_dev(batch, out_f, input->device, input->gpu_backend);
        l->a = alloc2_dev(batch, out_f, input->device, input->gpu_backend);
    }

#ifdef USE_CUDA
    if (CF_IS_CUDA(input)) {
        CF_CHECK(CF_IS_CUDA(l->W),
                 "dense_forward: layer weights are on CPU but input is on GPU "
                 "— call dense_to_gpu() first");
        cuda_linear_forward(input->data, l->W->data, l->b->data, l->z->data,
                             batch, in_f, out_f);
    } else
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(input)) {
        CF_CHECK(CF_IS_OPENCL(l->W),
                 "dense_forward: layer weights are on CPU but input is on GPU "
                 "— call dense_to_gpu() first");
        opencl_linear_forward(input->data, l->W->data, l->b->data, l->z->data,
                               batch, in_f, out_f);
    } else
#endif
    {
    CF_CHECK(input->device == CF_DEVICE_CPU,
             "dense_forward: input is GPU-resident on a backend this build "
             "doesn't support — recompile with -DUSE_CUDA/-DUSE_OPENCL or "
             "call tensor_to_cpu() first");
    /* z = input @ W^T  (W is [out,in], so W^T is [in,out])      */
    /* z[b,o] = sum_i input[b,i]*W[o,i] + b[o]                   */
    /* Parallel over batch — each b writes to different rows of z */
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(batch * out_f > 1024)
#endif
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_f; o++) {
            float acc = l->b->data[o];
            for (int i = 0; i < in_f; i++)
                acc += input->data[b*in_f + i] * l->W->data[o*in_f + i];
            l->z->data[b*out_f + o] = acc;
        }
    }
    }

    /* activation */
    switch (l->activation) {
        case ACT_RELU:    tensor_relu(l->z, l->a);    break;
        case ACT_SIGMOID: tensor_sigmoid(l->z, l->a); break;
        case ACT_TANH:    tensor_tanh_t(l->z, l->a);  break;
        case ACT_SOFTMAX: tensor_softmax(l->z, l->a); break;
        default:          tensor_copy_data(l->a, l->z); break;
    }

    tensor_copy_data(output, l->a);
}

/* ── backward pass ───────────────────────────────────────────────── */
/*
 * grad_out: upstream gradient dL/da  [batch, out_f]
 *
 * 1. dL/dz  via activation derivative
 * 2. dL/dW  = dz^T @ input          [out_f, in_f]
 * 3. dL/db  = sum over batch of dz   [out_f]
 * 4. dL/dX  = dz @ W                [batch, in_f]
 */
void dense_backward(DenseLayer *l, const Tensor *grad_out) {
    int batch = l->input->shape[0];
    int in_f  = l->in_features;
    int out_f = l->out_features;

    /* allocate dz (same shape as z, same device+backend as l->z) */
    Tensor *dz = alloc2_dev(batch, out_f, l->z->device, l->z->gpu_backend);

    /* activation gradient */
    switch (l->activation) {
        case ACT_RELU:
            tensor_relu_grad(l->z, grad_out, dz);
            break;
        case ACT_SIGMOID:
            tensor_sigmoid_grad(l->a, grad_out, dz);
            break;
        case ACT_TANH:
            tensor_tanh_grad(l->a, grad_out, dz);
            break;
        case ACT_SOFTMAX:
        case ACT_NONE:
        default:
            tensor_copy_data(dz, grad_out);
            break;
    }

    /* dX: allocate/resize on the same device+backend as dz first, since
     * the GPU path below needs it ready before the fused kernel call  */
    if (!l->dX || l->dX->shape[0] != batch || l->dX->device != dz->device ||
        l->dX->gpu_backend != dz->gpu_backend) {
        tensor_free(l->dX);
        l->dX = alloc2_dev(batch, in_f, dz->device, dz->gpu_backend);
    }

#ifdef USE_CUDA
    if (CF_IS_CUDA(dz)) {
        /* one fused call computes dW, db (already averaged by batch)
         * and dX directly on the GPU — see cuda_linear_backward()   */
        cuda_linear_backward(l->input->data, l->W->data, dz->data,
                              l->dW->data, l->db->data, l->dX->data,
                              batch, in_f, out_f);
        tensor_free(dz);
        return;
    }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(dz)) {
        opencl_linear_backward(l->input->data, l->W->data, dz->data,
                                l->dW->data, l->db->data, l->dX->data,
                                batch, in_f, out_f);
        tensor_free(dz);
        return;
    }
#endif
    CF_CHECK(dz->device == CF_DEVICE_CPU,
             "dense_backward: gradient is GPU-resident on a backend this "
             "build doesn't support — recompile with -DUSE_CUDA/-DUSE_OPENCL "
             "or call tensor_to_cpu() first");

    /* dW = dz^T @ input  [out_f, in_f]                            */
    /* Parallel over o — each o writes to different row of dW      */
    tensor_fill(l->dW, 0.0f);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(out_f * in_f > 1024)
#endif
    for (int o = 0; o < out_f; o++) {
        for (int b = 0; b < batch; b++) {
            float dzbo = dz->data[b*out_f + o];
            for (int i = 0; i < in_f; i++)
                l->dW->data[o*in_f + i] += dzbo * l->input->data[b*in_f + i];
        }
    }
    /* average over batch */
    tensor_scale(l->dW, 1.0f / batch, l->dW);

    /* db = mean over batch of dz  [out_f]                         */
    /* Parallel over o with local sum — no race condition           */
    tensor_fill(l->db, 0.0f);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(out_f > 64)
#endif
    for (int o = 0; o < out_f; o++) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++)
            sum += dz->data[b*out_f + o];
        l->db->data[o] = sum / (float)batch;
    }

    /* dX = dz @ W  [batch, in_f]                                  */
    /* Parallel over b — each b writes to different row of dX      */
    tensor_fill(l->dX, 0.0f);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(batch * in_f > 1024)
#endif
    for (int b = 0; b < batch; b++) {
        for (int o = 0; o < out_f; o++) {
            float dzbo = dz->data[b*out_f + o];
            for (int i = 0; i < in_f; i++)
                l->dX->data[b*in_f + i] += dzbo * l->W->data[o*in_f + i];
        }
    }

    tensor_free(dz);
}

int dense_param_count(const DenseLayer *l) {
    return l->in_features * l->out_features + l->out_features;
}
