#include "optimizer.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* NOTE ON THE HETEROGENEOUS-LAYER REFACTOR:
 * Network::layers[] used to be a plain DenseLayer* array, so every
 * optimizer here just walked it and touched ->W/->b/->dW/->db
 * directly. Now that layers can be Dense, Dropout, BatchNorm, RNN, or
 * LSTM, that no longer typechecks (and wouldn't be correct even if it
 * did — the other four modules don't have W/b/dW/db).
 *
 * Only Dense actually needs Adam/RMSProp/momentum-SGD's per-parameter
 * moment buffers, so those remain keyed by layer index exactly as
 * before, but every step function now switches on nl->type and simply
 * skips non-Dense layers (Dropout has no parameters to update at
 * all). BatchNorm/RNN/LSTM already ship their own hand-written
 * plain-SGD updaters (batchnorm_update_sgd/rnn_update_sgd/
 * lstm_update_sgd) — those are dispatched alongside sgd_step() below
 * so a heterogeneous network trained with SGD still gets every layer
 * updated, just without momentum/weight-decay on the non-Dense ones
 * (those modules don't expose the hooks needed to add it). Adam and
 * RMSProp, as provided, only ever update Dense parameters — there's
 * no Adam/RMSProp variant of the other modules' updaters to call.
 */

/* ── helpers ─────────────────────────────────────────────────────── */

static Tensor *zeros_like(const Tensor *t) {
    return tensor_zeros(t->shape, t->ndim);
}

static DenseLayer *as_dense(const NetworkLayer *nl) {
    return nl->type == LAYER_DENSE ? (DenseLayer *)nl->layer_ptr : NULL;
}

/* ── SGD ─────────────────────────────────────────────────────────── */

SGD *sgd_create(float lr, float momentum, float weight_decay) {
    SGD *opt = (SGD *)calloc(1, sizeof(SGD));
    opt->lr           = lr;
    opt->momentum     = momentum;
    opt->weight_decay = weight_decay;
    return opt;
}

static void sgd_init(SGD *opt, Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;
        opt->vW[i] = zeros_like(l->W);
        opt->vb[i] = zeros_like(l->b);
    }
    opt->initialized = 1;
}

void sgd_step(SGD *opt, Network *net) {
    if (!opt->initialized) sgd_init(opt, net);

    for (int i = 0; i < net->num_layers; i++) {
        NetworkLayer *nl = &net->layers[i];

        switch (nl->type) {
        case LAYER_DENSE: {
            DenseLayer *l = (DenseLayer *)nl->layer_ptr;
            size_t nW = l->W->size;
            size_t nb = l->b->size;

            /* W update */
            for (size_t k = 0; k < nW; k++) {
                float g = l->dW->data[k] + opt->weight_decay * l->W->data[k];
                opt->vW[i]->data[k] = opt->momentum * opt->vW[i]->data[k]
                                       + (1.0f - opt->momentum) * g;
                l->W->data[k] -= opt->lr * opt->vW[i]->data[k];
            }
            /* b update (no weight decay on bias) */
            for (size_t k = 0; k < nb; k++) {
                float g = l->db->data[k];
                opt->vb[i]->data[k] = opt->momentum * opt->vb[i]->data[k]
                                       + (1.0f - opt->momentum) * g;
                l->b->data[k] -= opt->lr * opt->vb[i]->data[k];
            }
            break;
        }
        case LAYER_BATCHNORM:
            /* BatchNorm ships its own plain-SGD updater (no momentum/
             * weight-decay knobs available on it). */
            batchnorm_update_sgd((BatchNorm *)nl->layer_ptr, opt->lr);
            break;
        case LAYER_RNN:
            rnn_update_sgd((RNNLayer *)nl->layer_ptr, opt->lr);
            break;
        case LAYER_LSTM:
            lstm_update_sgd((LSTMLayer *)nl->layer_ptr, opt->lr);
            break;
        case LAYER_DROPOUT:
        default:
            break;   /* no learnable parameters */
        }
    }
}

void sgd_free(SGD *opt) {
    if (!opt) return;
    for (int i = 0; i < NN_MAX_LAYERS; i++) {
        tensor_free(opt->vW[i]);
        tensor_free(opt->vb[i]);
    }
    free(opt);
}

/* ── Adam ────────────────────────────────────────────────────────── */

Adam *adam_create(float lr, float beta1, float beta2, float eps,
                  float weight_decay) {
    Adam *opt = (Adam *)calloc(1, sizeof(Adam));
    opt->lr           = lr;
    opt->beta1        = beta1;
    opt->beta2        = beta2;
    opt->eps          = eps;
    opt->weight_decay = weight_decay;
    opt->t            = 0;
    return opt;
}

static void adam_init(Adam *opt, Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;
        opt->mW[i] = zeros_like(l->W);
        opt->vW[i] = zeros_like(l->W);
        opt->mb[i] = zeros_like(l->b);
        opt->vb[i] = zeros_like(l->b);
    }
    opt->initialized = 1;
}

void adam_step(Adam *opt, Network *net) {
    if (!opt->initialized) adam_init(opt, net);
    opt->t++;

    float b1  = opt->beta1, b2  = opt->beta2;
    float eps = opt->eps,   wd  = opt->weight_decay;
    float lr  = opt->lr;
    int   t   = opt->t;

    /* bias-corrected lr */
    float lr_t = lr * sqrtf(1.0f - powf(b2, t)) / (1.0f - powf(b1, t));

    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;   /* Adam only defined for Dense here — see file header note */

        /* weights */
        size_t nW = l->W->size;
        for (size_t k = 0; k < nW; k++) {
            float g = l->dW->data[k] + wd * l->W->data[k];
            opt->mW[i]->data[k] = b1 * opt->mW[i]->data[k] + (1.0f-b1) * g;
            opt->vW[i]->data[k] = b2 * opt->vW[i]->data[k] + (1.0f-b2) * g*g;
            l->W->data[k] -= lr_t * opt->mW[i]->data[k]
                             / (sqrtf(opt->vW[i]->data[k]) + eps);
        }

        /* biases */
        size_t nb = l->b->size;
        for (size_t k = 0; k < nb; k++) {
            float g = l->db->data[k];
            opt->mb[i]->data[k] = b1 * opt->mb[i]->data[k] + (1.0f-b1) * g;
            opt->vb[i]->data[k] = b2 * opt->vb[i]->data[k] + (1.0f-b2) * g*g;
            l->b->data[k] -= lr_t * opt->mb[i]->data[k]
                             / (sqrtf(opt->vb[i]->data[k]) + eps);
        }
    }
}

void adam_free(Adam *opt) {
    if (!opt) return;
    for (int i = 0; i < NN_MAX_LAYERS; i++) {
        tensor_free(opt->mW[i]); tensor_free(opt->vW[i]);
        tensor_free(opt->mb[i]); tensor_free(opt->vb[i]);
    }
    free(opt);
}

/* ── LR schedulers ───────────────────────────────────────────────── */

void lr_step_decay(SGD *opt, int epoch, int decay_every, float decay_rate) {
    if (epoch > 0 && epoch % decay_every == 0)
        opt->lr *= decay_rate;
}

void lr_cosine(SGD *opt, int epoch, int total_epochs,
               float lr_min, float lr_max) {
    opt->lr = lr_min + 0.5f * (lr_max - lr_min) *
              (1.0f + cosf((float)epoch / (float)total_epochs * 3.14159265f));
}

/* ── RMSProp ─────────────────────────────────────────────────────── */

RMSProp *rmsprop_create(float lr, float rho, float eps, float weight_decay) {
    RMSProp *opt = (RMSProp *)calloc(1, sizeof(RMSProp));
    opt->lr           = lr;
    opt->rho          = rho;
    opt->eps          = eps;
    opt->weight_decay = weight_decay;
    return opt;
}

static void rmsprop_init(RMSProp *opt, Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;
        opt->vW[i] = zeros_like(l->W);
        opt->vb[i] = zeros_like(l->b);
    }
    opt->initialized = 1;
}

void rmsprop_step(RMSProp *opt, Network *net) {
    if (!opt->initialized) rmsprop_init(opt, net);

    float rho = opt->rho, eps = opt->eps;
    float lr  = opt->lr,  wd  = opt->weight_decay;

    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;   /* RMSProp only defined for Dense here — see file header note */

        /* weights */
        for (size_t k = 0; k < l->W->size; k++) {
            float g = l->dW->data[k] + wd * l->W->data[k];
            opt->vW[i]->data[k] = rho * opt->vW[i]->data[k]
                                   + (1.0f - rho) * g * g;
            l->W->data[k] -= lr * g / (sqrtf(opt->vW[i]->data[k]) + eps);
        }

        /* biases */
        for (size_t k = 0; k < l->b->size; k++) {
            float g = l->db->data[k];
            opt->vb[i]->data[k] = rho * opt->vb[i]->data[k]
                                   + (1.0f - rho) * g * g;
            l->b->data[k] -= lr * g / (sqrtf(opt->vb[i]->data[k]) + eps);
        }
    }
}

void rmsprop_free(RMSProp *opt) {
    if (!opt) return;
    for (int i = 0; i < NN_MAX_LAYERS; i++) {
        tensor_free(opt->vW[i]);
        tensor_free(opt->vb[i]);
    }
    free(opt);
}

/* ── Gradient Clipping ───────────────────────────────────────────── */
/* Global L2 norm across every layer that has gradients this optimizer
 * module knows how to read directly (Dense's dW/db). BatchNorm/RNN/
 * LSTM keep their own dgamma/dbeta, dW_xh/dW_hh/db_h, dW_ih/dW_hh/db
 * — not included in this global norm/clip pass; clip those via their
 * own module if/when they expose a norm helper. */

float nn_grad_norm(const Network *net) {
    float total = 0.0f;
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;
        for (size_t k = 0; k < l->dW->size; k++)
            total += l->dW->data[k] * l->dW->data[k];
        for (size_t k = 0; k < l->db->size; k++)
            total += l->db->data[k] * l->db->data[k];
    }
    return sqrtf(total);
}

void nn_clip_gradients(Network *net, float max_norm) {
    float norm = nn_grad_norm(net);
    if (norm <= max_norm) return;   /* already within limit */

    float scale = max_norm / (norm + 1e-6f);
    for (int i = 0; i < net->num_layers; i++) {
        DenseLayer *l = as_dense(&net->layers[i]);
        if (!l) continue;
        for (size_t k = 0; k < l->dW->size; k++)
            l->dW->data[k] *= scale;
        for (size_t k = 0; k < l->db->size; k++)
            l->db->data[k] *= scale;
    }
}
