#include "conv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_backend.h"
#endif

/* ── index helpers ───────────────────────────────────────────────── */
/*
 * 4D tensor layout: [batch, channels, height, width]
 * index(b, c, h, w) = b*(C*H*W) + c*(H*W) + h*W + w
 */
#define IDX4(b,c,h,w, C,H,W) \
    ((size_t)(b)*(C)*(H)*(W) + (size_t)(c)*(H)*(W) + (size_t)(h)*(W) + (w))

/* ── lifecycle ───────────────────────────────────────────────────── */

Conv2D *conv2d_create(int in_channels, int out_channels,
                      int kernel_size, int stride, int pad) {
    Conv2D *l = (Conv2D *)calloc(1, sizeof(Conv2D));
    CF_CHECK_ALLOC(l);
    l->in_channels  = in_channels;
    l->out_channels = out_channels;
    l->kernel_h     = kernel_size;
    l->kernel_w     = kernel_size;
    l->stride       = stride;
    l->pad          = pad;
    l->training     = 1;

    /* W shape: [out_ch, in_ch, kH, kW] */
    int shW[4] = {out_channels, in_channels, kernel_size, kernel_size};
    int shb[1] = {out_channels};
    l->W  = tensor_zeros(shW, 4);
    l->b  = tensor_zeros(shb, 1);
    l->dW = tensor_zeros(shW, 4);
    l->db = tensor_zeros(shb, 1);

    conv2d_init_weights(l);
    return l;
}

void conv2d_free(Conv2D *l) {
    if (!l) return;
    tensor_free(l->W);  tensor_free(l->b);
    tensor_free(l->dW); tensor_free(l->db);
    tensor_free(l->input_cache);
    free(l);
}

void conv2d_init_weights(Conv2D *l) {
    /* He initialization: scale = sqrt(2 / fan_in) */
    int fan_in = l->in_channels * l->kernel_h * l->kernel_w;
    float scale = sqrtf(2.0f / (float)fan_in);
    int sh[4] = {l->out_channels, l->in_channels, l->kernel_h, l->kernel_w};
    Tensor *tmp = tensor_randn(sh, 4);
    tensor_scale(tmp, scale, l->W);
    tensor_free(tmp);
    tensor_fill(l->b, 0.0f);
}

/* ── output size ─────────────────────────────────────────────────── */

void conv2d_output_size(const Conv2D *l, int H, int W,
                        int *out_h, int *out_w) {
    *out_h = (H + 2*l->pad - l->kernel_h) / l->stride + 1;
    *out_w = (W + 2*l->pad - l->kernel_w) / l->stride + 1;
}

/* ── padding helper ──────────────────────────────────────────────── */

static Tensor *pad_input(const Tensor *input, int pad) {
    if (pad == 0) return tensor_clone(input);
    int batch = input->shape[0];
    int C     = input->shape[1];
    int H     = input->shape[2];
    int W     = input->shape[3];
    int pH    = H + 2*pad;
    int pW    = W + 2*pad;
    int sh[4] = {batch, C, pH, pW};

#ifdef USE_CUDA
    if (CF_IS_CUDA(input)) {
        Tensor *out = (Tensor *)malloc(sizeof(Tensor));
        CF_CHECK_ALLOC(out);
        out->ndim = 4;
        memcpy(out->shape, sh, sizeof(sh));
        out->size = (size_t)batch * C * pH * pW;
        out->owns_data = 1;
        out->device = CF_DEVICE_GPU;
        out->gpu_backend = CF_GPU_CUDA;
        out->data = cuda_malloc_f32(out->size);
        cuda_pad2d(input->data, out->data, batch, C, H, W, pad);
        return out;
    }
#endif
    CF_CHECK(input->device == CF_DEVICE_CPU,
             "conv2d: input is GPU-resident on a backend Conv2D doesn't "
             "support yet (CUDA only) — call tensor_to_cpu() first");

    Tensor *out = tensor_zeros(sh, 4);
    for (int b = 0; b < batch; b++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    out->data[IDX4(b,c,h+pad,w+pad, C,pH,pW)] =
                        input->data[IDX4(b,c,h,w, C,H,W)];
    return out;
}

/* ── GPU residency ───────────────────────────────────────────────── */

void conv2d_to_gpu(Conv2D *l) {
#ifndef USE_CUDA
    (void)l;
    cforge_error("conv2d_to_gpu: Conv2D/MaxPool2D GPU support is CUDA-only "
                 "for now (no OpenCL kernels yet) — library was built "
                 "without -DUSE_CUDA", __FILE__, __LINE__);
#else
    /* explicit CF_GPU_CUDA, NOT tensor_to_gpu() — if only USE_OPENCL is
     * compiled in, the generic picker would move these to OpenCL
     * buffers that conv2d_forward/backward can't read (no OpenCL conv
     * kernels exist yet), silently corrupting memory instead of
     * erroring. Fail loud and clear instead. */
    tensor_to_gpu_ex(l->W, CF_GPU_CUDA);
    tensor_to_gpu_ex(l->b, CF_GPU_CUDA);
    tensor_to_gpu_ex(l->dW, CF_GPU_CUDA);
    tensor_to_gpu_ex(l->db, CF_GPU_CUDA);
#endif
}

void conv2d_to_cpu(Conv2D *l) {
#ifndef USE_CUDA
    (void)l;
    cforge_error("conv2d_to_cpu: library was built without -DUSE_CUDA",
                 __FILE__, __LINE__);
#else
    tensor_to_cpu(l->W);
    tensor_to_cpu(l->b);
    tensor_to_cpu(l->dW);
    tensor_to_cpu(l->db);
    if (l->input_cache) tensor_to_cpu(l->input_cache);
#endif
}

/* ── forward ─────────────────────────────────────────────────────── */

void conv2d_forward(Conv2D *l, const Tensor *input, Tensor *output) {
    CF_CHECK(input->ndim == 4,
             "conv2d_forward: input must be 4D [batch,channels,H,W]");
    int batch = input->shape[0];
    int in_C  = input->shape[1];
    int in_H  = input->shape[2];
    int in_W  = input->shape[3];
    CF_CHECK(in_C == l->in_channels,
             "conv2d_forward: input channels != l->in_channels");

    int out_H, out_W;
    conv2d_output_size(l, in_H, in_W, &out_H, &out_W);

    int out_C = l->out_channels;
    int kH    = l->kernel_h;
    int kW    = l->kernel_w;
    int S     = l->stride;

    /* pad input and cache it for backward */
    tensor_free(l->input_cache);
    l->input_cache = pad_input(input, l->pad);
    l->batch_cache = batch;
    l->in_h_cache  = in_H;
    l->in_w_cache  = in_W;

    int pH = in_H + 2*l->pad;
    int pW = in_W + 2*l->pad;

    tensor_fill(output, 0.0f);

#ifdef USE_CUDA
    if (CF_IS_CUDA(input)) {
        CF_CHECK(CF_IS_CUDA(l->W),
                 "conv2d_forward: layer weights are on CPU but input is on "
                 "GPU — call conv2d_to_gpu() first");
        cuda_conv2d_forward(l->input_cache->data, l->W->data, l->b->data,
                             output->data,
                             batch, in_C, pH, pW, out_C, kH, kW, S,
                             out_H, out_W);
        return;
    }
#endif
    CF_CHECK(input->device == CF_DEVICE_CPU,
             "conv2d_forward: input is GPU-resident on a backend Conv2D "
             "doesn't support yet (CUDA only) — call tensor_to_cpu() first");

    /*
     * output[b, oc, oh, ow] =
     *   sum_{ic, kh, kw} W[oc,ic,kh,kw] * padded[b,ic, oh*S+kh, ow*S+kw]
     *   + b[oc]
     */
    for (int b  = 0; b  < batch; b++ )
    for (int oc = 0; oc < out_C; oc++)
    for (int oh = 0; oh < out_H; oh++ )
    for (int ow = 0; ow < out_W; ow++ ) {
        float acc = l->b->data[oc];
        for (int ic = 0; ic < in_C;  ic++)
        for (int kh = 0; kh < kH;    kh++)
        for (int kw = 0; kw < kW;    kw++) {
            int ih = oh*S + kh;
            int iw = ow*S + kw;
            acc += l->W->data[IDX4(oc,ic,kh,kw, in_C,kH,kW)]
                 * l->input_cache->data[IDX4(b,ic,ih,iw, in_C,pH,pW)];
        }
        output->data[IDX4(b,oc,oh,ow, out_C,out_H,out_W)] = acc;
    }
}

/* ── backward ────────────────────────────────────────────────────── */

void conv2d_backward(Conv2D *l, const Tensor *grad_out, Tensor *grad_in) {
    CF_CHECK(l->input_cache != NULL,
             "conv2d_backward: must call forward first");

    int batch = l->batch_cache;
    int in_C  = l->in_channels;
    int in_H  = l->in_h_cache;
    int in_W  = l->in_w_cache;
    int out_C = l->out_channels;
    int kH    = l->kernel_h;
    int kW    = l->kernel_w;
    int S     = l->stride;

    int out_H, out_W;
    conv2d_output_size(l, in_H, in_W, &out_H, &out_W);

    int pH = in_H + 2*l->pad;
    int pW = in_W + 2*l->pad;

    /* padded gradient buffer for input */
    int sh_pg[4] = {batch, in_C, pH, pW};
    Tensor *dpad;
#ifdef USE_CUDA
    if (CF_IS_CUDA(grad_out)) {
        CF_CHECK(CF_IS_CUDA(l->W) && CF_IS_CUDA(l->input_cache),
                 "conv2d_backward: layer/cache are on CPU but grad_out is on "
                 "GPU — call conv2d_to_gpu() first");
        dpad = (Tensor *)malloc(sizeof(Tensor));
        CF_CHECK_ALLOC(dpad);
        dpad->ndim = 4;
        memcpy(dpad->shape, sh_pg, sizeof(sh_pg));
        dpad->size = (size_t)batch * in_C * pH * pW;
        dpad->owns_data = 1;
        dpad->device = CF_DEVICE_GPU;
        dpad->gpu_backend = CF_GPU_CUDA;
        dpad->data = cuda_malloc_f32(dpad->size);

        cuda_conv2d_backward(l->input_cache->data, l->W->data, grad_out->data,
                              l->dW->data, l->db->data, dpad->data,
                              batch, in_C, pH, pW, out_C, kH, kW, S,
                              out_H, out_W);

        /* scale gradients by 1/batch (matches CPU path exactly) */
        tensor_scale(l->dW, 1.0f/(float)batch, l->dW);
        tensor_scale(l->db, 1.0f/(float)batch, l->db);

        cuda_unpad2d(dpad->data, grad_in->data, batch, in_C, in_H, in_W, l->pad);
        tensor_free(dpad);
        return;
    }
#endif
    CF_CHECK(grad_out->device == CF_DEVICE_CPU,
             "conv2d_backward: grad_out is GPU-resident on a backend Conv2D "
             "doesn't support yet (CUDA only) — call tensor_to_cpu() first");
    dpad = tensor_zeros(sh_pg, 4);

    tensor_fill(l->dW, 0.0f);
    tensor_fill(l->db, 0.0f);

    for (int b  = 0; b  < batch; b++ )
    for (int oc = 0; oc < out_C; oc++)
    for (int oh = 0; oh < out_H; oh++ )
    for (int ow = 0; ow < out_W; ow++ ) {
        float g = grad_out->data[IDX4(b,oc,oh,ow, out_C,out_H,out_W)];

        /* db */
        l->db->data[oc] += g;

        for (int ic = 0; ic < in_C; ic++)
        for (int kh = 0; kh < kH;   kh++)
        for (int kw = 0; kw < kW;   kw++) {
            int ih = oh*S + kh;
            int iw = ow*S + kw;
            /* dW */
            l->dW->data[IDX4(oc,ic,kh,kw, in_C,kH,kW)] +=
                g * l->input_cache->data[IDX4(b,ic,ih,iw, in_C,pH,pW)];
            /* dpadded input */
            dpad->data[IDX4(b,ic,ih,iw, in_C,pH,pW)] +=
                g * l->W->data[IDX4(oc,ic,kh,kw, in_C,kH,kW)];
        }
    }

    /* scale gradients by 1/batch */
    tensor_scale(l->dW, 1.0f/(float)batch, l->dW);
    tensor_scale(l->db, 1.0f/(float)batch, l->db);

    /* unpad dpad → grad_in */
    tensor_fill(grad_in, 0.0f);
    for (int b = 0; b < batch; b++)
    for (int c = 0; c < in_C;  c++)
    for (int h = 0; h < in_H;  h++)
    for (int w = 0; w < in_W;  w++)
        grad_in->data[IDX4(b,c,h,w, in_C,in_H,in_W)] =
            dpad->data[IDX4(b,c,h+l->pad,w+l->pad, in_C,pH,pW)];

    tensor_free(dpad);
}

/* ── MaxPool2D ───────────────────────────────────────────────────── */

void maxpool2d_forward(const Tensor *input, Tensor *output, Tensor *mask,
                       int pool_size, int stride) {
    CF_CHECK(input->ndim == 4,
             "maxpool2d_forward: input must be 4D [batch,ch,H,W]");
    int batch = input->shape[0];
    int C     = input->shape[1];
    int H     = input->shape[2];
    int W     = input->shape[3];
    int out_H = (H - pool_size) / stride + 1;
    int out_W = (W - pool_size) / stride + 1;

#ifdef USE_CUDA
    if (CF_IS_CUDA(input)) {
        cuda_maxpool2d_forward(input->data, output->data, mask->data,
                                batch, C, H, W, pool_size, stride,
                                out_H, out_W);
        return;
    }
#endif
    CF_CHECK(input->device == CF_DEVICE_CPU,
             "maxpool2d_forward: input is GPU-resident on a backend "
             "MaxPool2D doesn't support yet (CUDA only) — call "
             "tensor_to_cpu() first");

    for (int b = 0; b < batch; b++)
    for (int c = 0; c < C;     c++)
    for (int oh = 0; oh < out_H; oh++)
    for (int ow = 0; ow < out_W; ow++) {
        float max_val = -1e38f;
        int   max_idx = 0;
        for (int kh = 0; kh < pool_size; kh++)
        for (int kw = 0; kw < pool_size; kw++) {
            int ih = oh*stride + kh;
            int iw = ow*stride + kw;
            float v = input->data[IDX4(b,c,ih,iw, C,H,W)];
            if (v > max_val) {
                max_val = v;
                max_idx = (int)IDX4(b,c,ih,iw, C,H,W);
            }
        }
        size_t out_idx = IDX4(b,c,oh,ow, C,out_H,out_W);
        output->data[out_idx] = max_val;
        mask->data[out_idx]   = (float)max_idx;
    }
}

void maxpool2d_backward(const Tensor *grad_out, Tensor *grad_in,
                        const Tensor *mask, int pool_size, int stride) {
#ifdef USE_CUDA
    if (CF_IS_CUDA(grad_out)) {
        int batch = grad_in->shape[0], C = grad_in->shape[1];
        int H     = grad_in->shape[2], W = grad_in->shape[3];
        int out_H = grad_out->shape[2], out_W = grad_out->shape[3];
        cuda_maxpool2d_backward(grad_out->data, grad_in->data, mask->data,
                                 batch, C, H, W, pool_size, stride,
                                 out_H, out_W);
        return;
    }
#endif
    CF_CHECK(grad_out->device == CF_DEVICE_CPU,
             "maxpool2d_backward: grad_out is GPU-resident on a backend "
             "MaxPool2D doesn't support yet (CUDA only) — call "
             "tensor_to_cpu() first");
    (void)pool_size; (void)stride;
    tensor_fill(grad_in, 0.0f);
    for (size_t i = 0; i < grad_out->size; i++) {
        int src_idx = (int)mask->data[i];
        grad_in->data[src_idx] += grad_out->data[i];
    }
}

/* ── flatten ─────────────────────────────────────────────────────── */

Tensor *flatten(Tensor *t) {
    int batch    = t->shape[0];
    size_t feats = t->size / (size_t)batch;
    int new_sh[2] = {batch, (int)feats};
    return tensor_reshape(t, new_sh, 2);
}

/* ── param count ─────────────────────────────────────────────────── */

int conv2d_param_count(const Conv2D *l) {
    return (int)l->W->size + l->out_channels;
}
