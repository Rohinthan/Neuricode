#ifndef CONV_H
#define CONV_H

#include "tensor.h"

/*
 * Conv2D layer — 2D convolution for image inputs.
 *
 * Input  shape: [batch, in_channels,  height,      width     ]
 * Filter shape: [out_channels, in_channels, kernel_h, kernel_w]
 * Output shape: [batch, out_channels, out_h,       out_w     ]
 *
 * out_h = (height + 2*pad - kernel_h) / stride + 1
 * out_w = (width  + 2*pad - kernel_w) / stride + 1
 *
 * Supported: same padding (pad auto-calculated) or manual pad.
 */

typedef struct {
    /* hyperparameters */
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride;
    int pad;        /* zero-padding on each side */
    int training;

    /* parameters */
    Tensor *W;      /* filters [out_ch, in_ch, kH, kW] */
    Tensor *b;      /* bias    [out_ch]                 */

    /* gradients */
    Tensor *dW;
    Tensor *db;

    /* cache for backward */
    Tensor *input_cache;   /* padded input [batch, in_ch, H+2p, W+2p] */
    int     batch_cache;
    int     in_h_cache;
    int     in_w_cache;
} Conv2D;

/* ── lifecycle ──────────────────────────────────────────────────── */
/*
 * conv2d_create:
 *   in_channels   number of input channels  (e.g. 1 for grayscale, 3 for RGB)
 *   out_channels  number of filters
 *   kernel_size   square kernel (kernel_size x kernel_size)
 *   stride        convolution stride (1 = no skip)
 *   pad           zero-padding per side (0 = valid, 1 = same for stride=1)
 */
Conv2D *conv2d_create(int in_channels, int out_channels,
                      int kernel_size, int stride, int pad);
void    conv2d_free(Conv2D *l);

/* He initialization for conv weights */
void    conv2d_init_weights(Conv2D *l);

/* ── forward ────────────────────────────────────────────────────── */
/*
 * input  [batch, in_ch,  H,     W    ]  — flat row-major
 * output [batch, out_ch, out_h, out_w]  — caller allocates
 *
 * Helper to compute output spatial dims:
 */
void conv2d_output_size(const Conv2D *l, int H, int W,
                        int *out_h, int *out_w);

void conv2d_forward(Conv2D *l, const Tensor *input, Tensor *output);

/* ── backward ───────────────────────────────────────────────────── */
/*
 * grad_out [batch, out_ch, out_h, out_w]  upstream gradient
 * grad_in  [batch, in_ch,  H,     W    ]  grad w.r.t. input (caller allocs)
 * Also fills l->dW and l->db.
 */
void conv2d_backward(Conv2D *l, const Tensor *grad_out, Tensor *grad_in);

/* ── pooling (standalone, no learnable params) ──────────────────── */
/*
 * maxpool2d_forward:
 *   input  [batch, channels, H,  W ]
 *   output [batch, channels, H', W']  where H'=H/pool_size
 *   mask   [same shape as output] — stores argmax index for backward
 */
void maxpool2d_forward(const Tensor *input, Tensor *output, Tensor *mask,
                       int pool_size, int stride);

void maxpool2d_backward(const Tensor *grad_out, Tensor *grad_in,
                        const Tensor *mask, int pool_size, int stride);

/* ── flatten ────────────────────────────────────────────────────── */
/*
 * Reshape [batch, C, H, W] → [batch, C*H*W]
 * Returns a VIEW (no copy) — free with tensor_free() when done.
 */
Tensor *flatten(Tensor *t);

/* ── param count ────────────────────────────────────────────────── */
int conv2d_param_count(const Conv2D *l);

/* ── GPU residency ───────────────────────────────────────────────
 * Moves W, b, dW, db to/from the GPU. After conv2d_to_gpu(), pass
 * GPU-resident input tensors (see tensor_to_gpu()) to
 * conv2d_forward/backward and maxpool2d_forward/backward as usual.
 * Requires the library to be built with -DUSE_CUDA.               */
void conv2d_to_gpu(Conv2D *l);
void conv2d_to_cpu(Conv2D *l);

#endif /* CONV_H */
