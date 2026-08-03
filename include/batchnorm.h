#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "tensor.h"

/*
 * Batch Normalization layer (1D — for Dense layer output).
 *
 * Forward (train):
 *   mu    = mean(x, axis=0)          per feature
 *   var   = variance(x, axis=0)
 *   x_hat = (x - mu) / sqrt(var+eps)
 *   y     = gamma * x_hat + beta
 *   running_mean/var updated for inference
 *
 * Forward (inference):
 *   x_hat = (x - running_mean) / sqrt(running_var + eps)
 *   y     = gamma * x_hat + beta
 *
 * Backward:
 *   Gradients w.r.t. gamma, beta, and input x.
 */
typedef struct {
    int     num_features;   /* must match layer output width      */
    float   eps;            /* numerical stability e.g. 1e-5      */
    float   momentum;       /* for running stats e.g. 0.1         */
    int     training;       /* 1=train, 0=eval                    */

    /* learnable parameters */
    Tensor *gamma;          /* scale  [num_features]              */
    Tensor *beta;           /* shift  [num_features]              */

    /* gradients */
    Tensor *dgamma;
    Tensor *dbeta;

    /* running statistics (updated during training) */
    Tensor *running_mean;
    Tensor *running_var;

    /* cache for backward pass */
    Tensor *x_hat;          /* normalized input   [batch, feats]  */
    Tensor *batch_mean;     /* mean per feature   [num_features]  */
    Tensor *batch_var;      /* var  per feature   [num_features]  */
    Tensor *input_cache;    /* copy of input      [batch, feats]  */
} BatchNorm;

/* ── lifecycle ──────────────────────────────────────────────────── */
BatchNorm *batchnorm_create(int num_features, float eps, float momentum);
void       batchnorm_free(BatchNorm *bn);

/* ── mode ───────────────────────────────────────────────────────── */
void batchnorm_train(BatchNorm *bn);
void batchnorm_eval(BatchNorm *bn);

/* ── forward / backward ─────────────────────────────────────────── */
/*
 * input  [batch, num_features]
 * output [batch, num_features]  (caller allocates)
 */
void batchnorm_forward(BatchNorm *bn, const Tensor *input, Tensor *output);

/*
 * grad_out [batch, num_features]  upstream gradient
 * grad_in  [batch, num_features]  gradient w.r.t. input (caller allocates)
 * Also fills bn->dgamma and bn->dbeta.
 */
void batchnorm_backward(BatchNorm *bn, const Tensor *grad_out, Tensor *grad_in);

/* ── parameter update (call after optimizer step) ───────────────── */
void batchnorm_update_sgd(BatchNorm *bn, float lr);
void batchnorm_zero_grad(BatchNorm *bn);

#endif /* BATCHNORM_H */
