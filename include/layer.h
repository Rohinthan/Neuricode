#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

/* ── activation enum ────────────────────────────────────────────── */
typedef enum {
    ACT_NONE    = 0,
    ACT_RELU    = 1,
    ACT_SIGMOID = 2,
    ACT_TANH    = 3,
    ACT_SOFTMAX = 4
} Activation;

/* ── Dense (fully-connected) layer ──────────────────────────────── */
typedef struct {
    int        in_features;
    int        out_features;
    Activation activation;

    Tensor    *W;       /* weights  [out, in]  */
    Tensor    *b;       /* bias     [out]      */

    /* cached forward values (set during forward pass) */
    Tensor    *input;   /* [batch, in]   — view, not owned */
    Tensor    *z;       /* pre-activation  [batch, out]    */
    Tensor    *a;       /* post-activation [batch, out]    */

    /* gradients (allocated on first backward) */
    Tensor    *dW;
    Tensor    *db;
    Tensor    *dX;      /* gradient w.r.t. input           */
} DenseLayer;

/* ── lifecycle ──────────────────────────────────────────────────── */
DenseLayer *dense_create(int in_features, int out_features, Activation act);
void        dense_free(DenseLayer *l);

/*  He init for ReLU, Xavier for others */
void        dense_init_weights(DenseLayer *l);

/* ── forward / backward ─────────────────────────────────────────── */
/*
 * forward:
 *   input  [batch, in_features]
 *   output [batch, out_features]   (caller allocates)
 */
void dense_forward(DenseLayer *l, const Tensor *input, Tensor *output);

/*
 * backward:
 *   grad_out  [batch, out_features]  gradient from upstream
 *   After call, l->dW, l->db, l->dX are populated.
 */
void dense_backward(DenseLayer *l, const Tensor *grad_out);

/* ── parameter access ───────────────────────────────────────────── */
int  dense_param_count(const DenseLayer *l);

/* ── GPU residency ───────────────────────────────────────────────
 * Moves W, b, dW, db (and any cached z/a/dX) to/from the GPU.
 * After dense_to_gpu(), pass GPU-resident input tensors (see
 * tensor_to_gpu()) to dense_forward/dense_backward as usual — the
 * same functions run either the CPU or CUDA path automatically.
 * Requires the library to be built with -DUSE_CUDA.               */
void dense_to_gpu(DenseLayer *l);
void dense_to_gpu_ex(DenseLayer *l, CF_GpuBackend backend);
void dense_to_cpu(DenseLayer *l);

#endif /* LAYER_H */
