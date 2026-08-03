#include "dropout.h"
#include <stdlib.h>
#include <time.h>

/* ── lifecycle ───────────────────────────────────────────────────── */

DropoutLayer *dropout_create(float drop_prob) {
    CF_CHECK(drop_prob >= 0.0f && drop_prob < 1.0f,
             "dropout_create: drop_prob must be in [0.0, 1.0)");
    DropoutLayer *l = (DropoutLayer *)calloc(1, sizeof(DropoutLayer));
    CF_CHECK_ALLOC(l);
    l->drop_prob = drop_prob;
    l->keep_prob = 1.0f - drop_prob;
    l->training  = 1;   /* training mode by default */
    l->mask      = NULL;
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    return l;
}

void dropout_free(DropoutLayer *l) {
    if (!l) return;
    tensor_free(l->mask);
    free(l);
}

/* ── mode ────────────────────────────────────────────────────────── */

void dropout_train(DropoutLayer *l) { l->training = 1; }
void dropout_eval(DropoutLayer *l)  { l->training = 0; }

/* ── forward ─────────────────────────────────────────────────────── */

void dropout_forward(DropoutLayer *l, const Tensor *input, Tensor *output) {
    CF_CHECK_SHAPE(input, output);

    /* inference: pass through unchanged */
    if (!l->training || l->drop_prob == 0.0f) {
        tensor_copy_data(output, input);
        return;
    }

    /* allocate / resize mask if needed */
    if (!l->mask || l->mask->size != input->size) {
        tensor_free(l->mask);
        l->mask = tensor_create(input->shape, input->ndim);
        CF_CHECK_ALLOC(l->mask);
    }

    /* inverted dropout: keep neurons with prob keep_prob,
       scale by 1/keep_prob so expected value stays the same */
    float scale = 1.0f / l->keep_prob;
    for (size_t i = 0; i < input->size; i++) {
        float keep = ((float)rand() / RAND_MAX) >= l->drop_prob ? 1.0f : 0.0f;
        l->mask->data[i]   = keep;
        output->data[i]    = input->data[i] * keep * scale;
    }
}

/* ── backward ────────────────────────────────────────────────────── */

void dropout_backward(DropoutLayer *l, const Tensor *grad_out,
                      Tensor *grad_in) {
    CF_CHECK_SHAPE(grad_out, grad_in);

    /* inference or no drop: gradient passes straight through */
    if (!l->training || l->drop_prob == 0.0f || !l->mask) {
        tensor_copy_data(grad_in, grad_out);
        return;
    }

    float scale = 1.0f / l->keep_prob;
    for (size_t i = 0; i < grad_out->size; i++)
        grad_in->data[i] = grad_out->data[i] * l->mask->data[i] * scale;
}
