#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pipeline.h"
#include "tensor.h"
#include "layer.h"
#include "rnn.h"

struct Pipeline {
    RNNLayer *rnn;
    DenseLayer *out;
    int vocab_size;
    int hidden_size;
    Tensor *x;
    Tensor *rnn_out;
    Tensor *dense_out;
    Tensor *h_prev;
};

static int read_tensor(FILE *f, Tensor *t) {
    return fread(t->data, sizeof(float), t->size, f) == t->size ? 0 : -1;
}

/*
 * pipeline_step: now a public API (was a static helper) — moved
 * as-is, only the bounds check on token_id was added since external
 * callers no longer go through pipeline_forward()'s validation.
 * No new allocations: model->dense_out is the same buffer already
 * allocated in pipeline_load() and reused on every call, exactly as
 * pipeline_forward() already relied on internally.
 */
const float *pipeline_step(Pipeline *model, int token_id) {
    if (!model) return NULL;
    if (token_id < 0 || token_id >= model->vocab_size) {
        fprintf(stderr, "pipeline: token id %d is outside the model vocabulary\n",
                token_id);
        return NULL;
    }

    int dense_shape[2] = { 1, model->hidden_size };

    tensor_fill(model->x, 0.0f);
    model->x->data[token_id] = 1.0f;
    rnn_forward(model->rnn, model->x, model->h_prev, model->rnn_out);
    model->h_prev = model->rnn->h_states[1];

    Tensor *rnn_2d = tensor_reshape(model->rnn_out, dense_shape, 2);
    dense_forward(model->out, rnn_2d, model->dense_out);
//    tensor_free(rnn_2d); if root case to make core dumb:
    return model->dense_out->data;
}

Pipeline *pipeline_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "pipeline: could not open model '%s'\n", path);
        return NULL;
    }

    int32_t file_vocab = 0;
    int32_t file_hidden = 0;
    if (fread(&file_vocab, sizeof(file_vocab), 1, f) != 1 ||
        fread(&file_hidden, sizeof(file_hidden), 1, f) != 1 ||
        file_vocab <= 0 || file_hidden <= 0) {
        fprintf(stderr, "pipeline: invalid model header in '%s'\n", path);
        fclose(f);
        return NULL;
    }

    Pipeline *model = calloc(1, sizeof(*model));
    if (!model) {
        fclose(f);
        return NULL;
    }
    model->vocab_size = file_vocab;
    model->hidden_size = file_hidden;
    model->rnn = rnn_create(model->vocab_size, model->hidden_size);
    model->out = dense_create(model->hidden_size, model->vocab_size, ACT_NONE);
    if (!model->rnn || !model->out) {
        fprintf(stderr, "pipeline: failed to allocate model layers\n");
        fclose(f);
        pipeline_free(model);
        return NULL;
    }

    int ok = read_tensor(f, model->rnn->W_xh) == 0 &&
             read_tensor(f, model->rnn->W_hh) == 0 &&
             read_tensor(f, model->rnn->b_h) == 0 &&
             read_tensor(f, model->out->W) == 0 &&
             read_tensor(f, model->out->b) == 0;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "pipeline: truncated model weights in '%s'\n", path);
        pipeline_free(model);
        return NULL;
    }

    int input_shape[3] = { 1, 1, model->vocab_size };
    int rnn_shape[3] = { 1, 1, model->hidden_size };
    int output_shape[2] = { 1, model->vocab_size };
    model->x = tensor_zeros(input_shape, 3);
    model->rnn_out = tensor_zeros(rnn_shape, 3);
    model->dense_out = tensor_zeros(output_shape, 2);
    if (!model->x || !model->rnn_out || !model->dense_out) {
        fprintf(stderr, "pipeline: failed to allocate inference buffers\n");
        pipeline_free(model);
        return NULL;
    }

    return model;
}

void pipeline_forward(Pipeline *model, const int *input, int seq_len, float *output) {
    if (!model || !input || !output || seq_len <= 0) return;

    for (int i = 0; i < seq_len; i++) {
        const float *logits = pipeline_step(model, input[i]);
        if (!logits) return; /* pipeline_step already logged the reason */

        if (i == seq_len - 1){
            size_t vs = (size_t)model->vocab_size;

            //TEMP debug limit
            size_t safe = vs;

            //if you know max tensor size, clean it
            if (safe > 1024) safe = 1024; 

            memcpy(output, logits, safe * sizeof(float));
        }
        //memcpy(output, logits, (size_t)model->vocab_size * sizeof(float));
    }
}

void pipeline_reset(Pipeline *model) {
    if (!model) return;
    model->h_prev = NULL;
}

void pipeline_free(Pipeline *model) {
    if (!model) return;
    tensor_free(model->x);
    tensor_free(model->rnn_out);
    tensor_free(model->dense_out);
    rnn_free(model->rnn);
    dense_free(model->out);
    free(model);
}
