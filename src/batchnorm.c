#include "batchnorm.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ── helpers ─────────────────────────────────────────────────────── */

static Tensor *alloc1(int n) {
    int sh[1] = {n};
    return tensor_zeros(sh, 1);
}
static Tensor *alloc2(int r, int c) {
    int sh[2] = {r, c};
    return tensor_zeros(sh, 2);
}

/* ── lifecycle ───────────────────────────────────────────────────── */

BatchNorm *batchnorm_create(int num_features, float eps, float momentum) {
    BatchNorm *bn = (BatchNorm *)calloc(1, sizeof(BatchNorm));
    CF_CHECK_ALLOC(bn);
    bn->num_features = num_features;
    bn->eps          = eps;
    bn->momentum     = momentum;
    bn->training     = 1;

    /* gamma allocated below */
    bn->beta         = alloc1(num_features);
    bn->dgamma       = alloc1(num_features);
    bn->dbeta        = alloc1(num_features);
    bn->running_mean = alloc1(num_features);
    /* running_var allocated below */

    /* gamma = ones, running_var = ones */
    int sh[1] = {num_features};
    tensor_free(bn->gamma);
    tensor_free(bn->running_var);
    bn->gamma       = tensor_ones(sh, 1);
    bn->running_var = tensor_ones(sh, 1);

    return bn;
}

void batchnorm_free(BatchNorm *bn) {
    if (!bn) return;
    tensor_free(bn->gamma);      tensor_free(bn->beta);
    tensor_free(bn->dgamma);     tensor_free(bn->dbeta);
    tensor_free(bn->running_mean);
    tensor_free(bn->running_var);
    tensor_free(bn->x_hat);
    tensor_free(bn->batch_mean);
    tensor_free(bn->batch_var);
    tensor_free(bn->input_cache);
    free(bn);
}

/* ── mode ────────────────────────────────────────────────────────── */

void batchnorm_train(BatchNorm *bn) { bn->training = 1; }
void batchnorm_eval(BatchNorm *bn)  { bn->training = 0; }

/* ── forward ─────────────────────────────────────────────────────── */

void batchnorm_forward(BatchNorm *bn, const Tensor *input, Tensor *output) {
    CF_CHECK(input->ndim == 2, "batchnorm_forward: input must be 2D [batch, features]");
    int batch = input->shape[0];
    int feats = input->shape[1];
    CF_CHECK(feats == bn->num_features,
             "batchnorm_forward: input features != bn->num_features");

    /* allocate caches if needed */
    if (!bn->x_hat || bn->x_hat->shape[0] != batch) {
        tensor_free(bn->x_hat);
        tensor_free(bn->input_cache);
        bn->x_hat       = alloc2(batch, feats);
        bn->input_cache = alloc2(batch, feats);
    }
    if (!bn->batch_mean) bn->batch_mean = alloc1(feats);
    if (!bn->batch_var)  bn->batch_var  = alloc1(feats);

    /* cache input */
    tensor_copy_data(bn->input_cache, input);

    if (bn->training) {
        /* ── compute batch mean per feature ── */
        tensor_fill(bn->batch_mean, 0.0f);
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < feats; f++)
                bn->batch_mean->data[f] += input->data[b*feats + f];
        for (int f = 0; f < feats; f++)
            bn->batch_mean->data[f] /= (float)batch;

        /* ── compute batch variance per feature ── */
        tensor_fill(bn->batch_var, 0.0f);
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < feats; f++) {
                float d = input->data[b*feats+f] - bn->batch_mean->data[f];
                bn->batch_var->data[f] += d * d;
            }
        for (int f = 0; f < feats; f++)
            bn->batch_var->data[f] /= (float)batch;

        /* ── update running stats ── */
        float m = bn->momentum;
        for (int f = 0; f < feats; f++) {
            bn->running_mean->data[f] =
                (1.0f-m)*bn->running_mean->data[f] + m*bn->batch_mean->data[f];
            bn->running_var->data[f]  =
                (1.0f-m)*bn->running_var->data[f]  + m*bn->batch_var->data[f];
        }

        /* ── normalize ── */
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < feats; f++) {
                float xhat = (input->data[b*feats+f] - bn->batch_mean->data[f])
                             / sqrtf(bn->batch_var->data[f] + bn->eps);
                bn->x_hat->data[b*feats+f]  = xhat;
                output->data[b*feats+f]      =
                    bn->gamma->data[f] * xhat + bn->beta->data[f];
            }

    } else {
        /* ── inference: use running stats ── */
        for (int b = 0; b < batch; b++)
            for (int f = 0; f < feats; f++) {
                float xhat = (input->data[b*feats+f] - bn->running_mean->data[f])
                             / sqrtf(bn->running_var->data[f] + bn->eps);
                output->data[b*feats+f] =
                    bn->gamma->data[f] * xhat + bn->beta->data[f];
            }
    }
}

/* ── backward ────────────────────────────────────────────────────── */

void batchnorm_backward(BatchNorm *bn, const Tensor *grad_out,
                        Tensor *grad_in) {
    int batch = bn->input_cache->shape[0];
    int feats = bn->num_features;
    float N   = (float)batch;

    /* dgamma = sum over batch of (grad_out * x_hat) */
    tensor_fill(bn->dgamma, 0.0f);
    for (int b = 0; b < batch; b++)
        for (int f = 0; f < feats; f++)
            bn->dgamma->data[f] +=
                grad_out->data[b*feats+f] * bn->x_hat->data[b*feats+f];

    /* dbeta = sum over batch of grad_out */
    tensor_fill(bn->dbeta, 0.0f);
    for (int b = 0; b < batch; b++)
        for (int f = 0; f < feats; f++)
            bn->dbeta->data[f] += grad_out->data[b*feats+f];

    /*
     * dL/dx = (1/N) * gamma / sqrt(var+eps) *
     *   (N*dout - sum(dout) - x_hat * sum(dout * x_hat))
     */
    for (int f = 0; f < feats; f++) {
        float inv_std = 1.0f / sqrtf(bn->batch_var->data[f] + bn->eps);
        float sum_dout = 0.0f, sum_dout_xhat = 0.0f;
        for (int b = 0; b < batch; b++) {
            sum_dout      += grad_out->data[b*feats+f];
            sum_dout_xhat += grad_out->data[b*feats+f] * bn->x_hat->data[b*feats+f];
        }
        for (int b = 0; b < batch; b++) {
            float dout = grad_out->data[b*feats+f];
            float xhat = bn->x_hat->data[b*feats+f];
            grad_in->data[b*feats+f] =
                (bn->gamma->data[f] * inv_std / N) *
                (N * dout - sum_dout - xhat * sum_dout_xhat);
        }
    }
}

/* ── parameter update ────────────────────────────────────────────── */

void batchnorm_update_sgd(BatchNorm *bn, float lr) {
    for (int f = 0; f < bn->num_features; f++) {
        bn->gamma->data[f] -= lr * bn->dgamma->data[f];
        bn->beta->data[f]  -= lr * bn->dbeta->data[f];
    }
}

void batchnorm_zero_grad(BatchNorm *bn) {
    tensor_fill(bn->dgamma, 0.0f);
    tensor_fill(bn->dbeta,  0.0f);
}
