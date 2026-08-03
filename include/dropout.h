#ifndef DROPOUT_H
#define DROPOUT_H

#include "tensor.h"

/*
 * Dropout layer — randomly zeros neurons during training.
 *
 * Uses inverted dropout so no scaling is needed at inference:
 *   forward (train):  mask random neurons, scale survivors by 1/keep_prob
 *   forward (infer):  pass through unchanged
 *   backward:         pass gradient only through surviving neurons
 */
typedef struct {
    float   drop_prob;   /* probability of DROPPING a neuron e.g. 0.5  */
    float   keep_prob;   /* 1.0 - drop_prob                             */
    int     training;    /* 1 = training mode, 0 = inference mode       */
    Tensor *mask;        /* binary mask from last forward pass           */
} DropoutLayer;

/* ── lifecycle ──────────────────────────────────────────────────── */
DropoutLayer *dropout_create(float drop_prob);
void          dropout_free(DropoutLayer *l);

/* ── mode ───────────────────────────────────────────────────────── */
void dropout_train(DropoutLayer *l);   /* switch to training mode  */
void dropout_eval(DropoutLayer *l);    /* switch to inference mode  */

/* ── forward / backward ─────────────────────────────────────────── */
/*
 * dropout_forward:
 *   input  — any shape tensor
 *   output — same shape as input (caller allocates)
 */
void dropout_forward(DropoutLayer *l, const Tensor *input, Tensor *output);

/*
 * dropout_backward:
 *   grad_out — upstream gradient (same shape as input)
 *   grad_in  — gradient w.r.t. input (caller allocates, same shape)
 */
void dropout_backward(DropoutLayer *l, const Tensor *grad_out, Tensor *grad_in);

#endif /* DROPOUT_H */
