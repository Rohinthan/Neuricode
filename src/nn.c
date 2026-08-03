#include "nn.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* NOTE ON ERROR HANDLING: CF_CHECK/cforge_error (from tensor.h) are
 * used instead of <assert.h> for structural invariants — same
 * convention already established in tensor.c/layer.c/batchnorm.c.
 * CF_CHECK reports a clear message + file/line and stays active in
 * release (NDEBUG) builds, where a plain assert() would compile away. */

/* ── lifecycle ───────────────────────────────────────────────────── */

Network *nn_create(void) {
    Network *net = (Network *)calloc(1, sizeof(Network));
    return net;
}

void nn_add_layer(Network *net, LayerType type, void *layer_ptr) {
    CF_CHECK(net->num_layers < NN_MAX_LAYERS,
             "nn_add_layer: exceeded NN_MAX_LAYERS — raise the limit or split the network");
    net->layers[net->num_layers].type      = type;
    net->layers[net->num_layers].layer_ptr = layer_ptr;
    net->num_layers++;
}

void nn_add_dense(Network *net, DenseLayer *layer)     { nn_add_layer(net, LAYER_DENSE,     layer); }
void nn_add_dropout(Network *net, DropoutLayer *layer) { nn_add_layer(net, LAYER_DROPOUT,   layer); }
void nn_add_batchnorm(Network *net, BatchNorm *layer)  { nn_add_layer(net, LAYER_BATCHNORM, layer); }
void nn_add_rnn(Network *net, RNNLayer *layer)         { nn_add_layer(net, LAYER_RNN,       layer); }
void nn_add_lstm(Network *net, LSTMLayer *layer)       { nn_add_layer(net, LAYER_LSTM,      layer); }

/* Frees every layer currently registered on `net` (dispatching to
 * each module's native destructor by type) plus its cached output/
 * grad-scratch buffers, then resets it to an empty-but-valid state.
 * Used by nn_free() and also by nn_load_model() (which needs to
 * discard a stale/previous model — or unwind a partially-constructed
 * one on read failure — without freeing `net` itself). */
static void nn_clear_layers(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        NetworkLayer *nl = &net->layers[i];
        switch (nl->type) {
        case LAYER_DENSE:     dense_free((DenseLayer *)nl->layer_ptr);     break;
        case LAYER_DROPOUT:   dropout_free((DropoutLayer *)nl->layer_ptr); break;
        case LAYER_BATCHNORM: batchnorm_free((BatchNorm *)nl->layer_ptr);  break;
        case LAYER_RNN:       rnn_free((RNNLayer *)nl->layer_ptr);         break;
        case LAYER_LSTM:      lstm_free((LSTMLayer *)nl->layer_ptr);       break;
        default:
            /* Unknown/corrupt type tag — nothing safe to cast+free,
             * just drop the pointer (better a leak than a bad free). */
            break;
        }
        tensor_free(net->outputs[i]);
        tensor_free(net->grad_cache[i]);
        net->outputs[i]          = NULL;
        net->grad_cache[i]       = NULL;
        net->layers[i].layer_ptr = NULL;
        net->layers[i].type      = (LayerType)0;
    }
    net->num_layers = 0;
}

void nn_free(Network *net) {
    if (!net) return;
    nn_clear_layers(net);
    free(net);
}

/* ── forward ─────────────────────────────────────────────────────── */

/* (Re)allocates *slot to the given shape only when it doesn't already
 * match (first call, or a shape change e.g. new batch size) — mirrors
 * the lazy-allocation behavior every layer module already uses for
 * its own internal caches (l->z/l->a, bn->x_hat, etc). */
static void ensure_buffer_shape(Tensor **slot, const int *shape, int ndim) {
    Tensor *cur = *slot;
    int match = cur && cur->ndim == ndim;
    if (match) {
        for (int d = 0; d < ndim; d++) {
            if (cur->shape[d] != shape[d]) { match = 0; break; }
        }
    }
    if (!match) {
        tensor_free(cur);
        *slot = tensor_zeros(shape, ndim);
        CF_CHECK_ALLOC(*slot);
    }
}

void nn_forward(Network *net, const Tensor *input, Tensor *output, int training) {
    const Tensor *cur = input;

    for (int i = 0; i < net->num_layers; i++) {
        NetworkLayer *nl = &net->layers[i];

        switch (nl->type) {
        case LAYER_DENSE: {
            DenseLayer *l = (DenseLayer *)nl->layer_ptr;
            CF_CHECK(cur->ndim == 2, "nn_forward: Dense expects 2D [batch, in_features] input");
            int sh[2] = { cur->shape[0], l->out_features };
            ensure_buffer_shape(&net->outputs[i], sh, 2);
            dense_forward(l, cur, net->outputs[i]);
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayer *l = (DropoutLayer *)nl->layer_ptr;
            /* Dropout is shape-agnostic and stores no dimensions of
             * its own; mode is persistent state, not a forward() arg. */
            if (training) dropout_train(l); else dropout_eval(l);
            ensure_buffer_shape(&net->outputs[i], cur->shape, cur->ndim);
            dropout_forward(l, cur, net->outputs[i]);
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNorm *l = (BatchNorm *)nl->layer_ptr;
            CF_CHECK(cur->ndim == 2, "nn_forward: BatchNorm expects 2D [batch, num_features] input");
            if (training) batchnorm_train(l); else batchnorm_eval(l);
            ensure_buffer_shape(&net->outputs[i], cur->shape, cur->ndim);
            batchnorm_forward(l, cur, net->outputs[i]);
            break;
        }
        case LAYER_RNN: {
            RNNLayer *l = (RNNLayer *)nl->layer_ptr;
            CF_CHECK(cur->ndim == 3, "nn_forward: RNN expects 3D [batch, seq_len, input_size] input");
            int sh[3] = { cur->shape[0], cur->shape[1], l->hidden_size };
            ensure_buffer_shape(&net->outputs[i], sh, 3);
            /* h_init = NULL: each nn_forward() call starts from a zero
             * hidden state (see the statelessness note in nn.h). */
            rnn_forward(l, cur, NULL, net->outputs[i]);
            break;
        }
        case LAYER_LSTM: {
            LSTMLayer *l = (LSTMLayer *)nl->layer_ptr;
            CF_CHECK(cur->ndim == 3, "nn_forward: LSTM expects 3D [batch, seq_len, input_size] input");
            int sh[3] = { cur->shape[0], cur->shape[1], l->hidden_size };
            ensure_buffer_shape(&net->outputs[i], sh, 3);
            /* h_init = c_init = NULL: zero initial hidden/cell state
             * every call (see the statelessness note in nn.h). */
            lstm_forward(l, cur, NULL, NULL, net->outputs[i]);
            break;
        }
        default:
            cforge_error("nn_forward: unrecognized LayerType in network "
                         "(corrupted Network struct?)", __FILE__, __LINE__);
        }

        cur = net->outputs[i];
    }

    /* copy final layer output to caller's buffer */
    CF_CHECK(output->size == cur->size,
             "nn_forward: caller-provided output buffer size doesn't match "
             "the last layer's output size");
    tensor_copy_data(output, cur);
}

/* ── loss ────────────────────────────────────────────────────────── */

float nn_loss(LossType type, const Tensor *pred, const Tensor *target,
              Tensor *grad) {
    CF_CHECK(pred->size == target->size, "nn_loss: pred/target size mismatch");
    CF_CHECK(grad->size == pred->size,   "nn_loss: grad buffer size mismatch");
    size_t N = pred->size;
    float  loss = 0.0f;

    switch (type) {
    case LOSS_MSE:
        /* L = mean((pred - target)^2)
           dL/dpred = 2*(pred - target) / N               */
        for (size_t i = 0; i < N; i++) {
            float d = pred->data[i] - target->data[i];
            loss += d * d;
            grad->data[i] = 2.0f * d / (float)N;
        }
        loss /= (float)N;
        break;

    case LOSS_BINARY_CROSS:
        /* L = -mean(t*log(p) + (1-t)*log(1-p))
           dL/dp = -(t/p - (1-t)/(1-p)) / N               */
        for (size_t i = 0; i < N; i++) {
            float p = pred->data[i];
            float t = target->data[i];
            p = p < 1e-7f ? 1e-7f : (p > 1.0f-1e-7f ? 1.0f-1e-7f : p);
            loss -= t * logf(p) + (1.0f - t) * logf(1.0f - p);
            grad->data[i] = (-(t / p) + (1.0f - t) / (1.0f - p)) / (float)N;
        }
        loss /= (float)N;
        break;

    case LOSS_CROSS_ENTROPY: {
        /*
         * Softmax + Cross-Entropy combined gradient.
         * L = -mean(sum_c t_c * log(p_c))
         *
         * Combined gradient: dL/dz = (p - t) / batch
         * This is numerically stable and accounts for
         * the softmax Jacobian automatically.
         * Using -(t/p) alone causes exploding gradients!
         */
        int batch   = pred->shape[0];
        int classes = (int)(N / batch);
        for (int b = 0; b < batch; b++) {
            float row_loss = 0.0f;
            for (int c = 0; c < classes; c++) {
                float p = pred->data[b*classes + c];
                float t = target->data[b*classes + c];
                p = p < 1e-7f ? 1e-7f : p;
                row_loss -= t * logf(p);
                /* KEY FIX: combined softmax+CE gradient */
                grad->data[b*classes + c] = (p - t) / (float)batch;
            }
            loss += row_loss;
        }
        loss /= (float)batch;
        break;
    }
    }
    return loss;
}

/* ── backward ────────────────────────────────────────────────────── */

void nn_backward(Network *net, const Tensor *grad_out) {
    const Tensor *g = grad_out;

    for (int i = net->num_layers - 1; i >= 0; i--) {
        NetworkLayer *nl = &net->layers[i];

        switch (nl->type) {
        case LAYER_DENSE: {
            DenseLayer *l = (DenseLayer *)nl->layer_ptr;
            dense_backward(l, g);
            g = l->dX;
            break;
        }
        case LAYER_DROPOUT: {
            /* DropoutLayer has no dX of its own — backward() writes
             * into a caller-supplied grad_in, so we route it through
             * this layer's cached scratch buffer (shape-agnostic:
             * dropout's grad_in shape == its own output's shape, since
             * dropout never changes shape). */
            DropoutLayer *l = (DropoutLayer *)nl->layer_ptr;
            ensure_buffer_shape(&net->grad_cache[i],
                                 net->outputs[i]->shape, net->outputs[i]->ndim);
            dropout_backward(l, g, net->grad_cache[i]);
            g = net->grad_cache[i];
            break;
        }
        case LAYER_BATCHNORM: {
            /* Same story as Dropout: no dX, caller-supplied grad_in,
             * in-shape == out-shape so we can size it off outputs[i]. */
            BatchNorm *l = (BatchNorm *)nl->layer_ptr;
            ensure_buffer_shape(&net->grad_cache[i],
                                 net->outputs[i]->shape, net->outputs[i]->ndim);
            batchnorm_backward(l, g, net->grad_cache[i]);
            g = net->grad_cache[i];
            break;
        }
        case LAYER_RNN: {
            RNNLayer *l = (RNNLayer *)nl->layer_ptr;
            rnn_backward(l, g);
            g = l->dX;
            break;
        }
        case LAYER_LSTM: {
            LSTMLayer *l = (LSTMLayer *)nl->layer_ptr;
            lstm_backward(l, g);
            g = l->dX;
            break;
        }
        default:
            cforge_error("nn_backward: unrecognized LayerType in network "
                         "(corrupted Network struct?)", __FILE__, __LINE__);
        }
    }
}

/* ── convenience training step ───────────────────────────────────── */

float nn_train_step(Network *net, const Tensor *input, const Tensor *target,
                    LossType loss_type, Tensor *pred_buf, Tensor *grad_buf) {
    nn_forward(net, input, pred_buf, /*training=*/1);
    float loss = nn_loss(loss_type, pred_buf, target, grad_buf);
    nn_backward(net, grad_buf);
    return loss;
}

/* ── utilities ───────────────────────────────────────────────────── */

/* BatchNorm and Dropout have no *_param_count() of their own (Dropout
 * genuinely has zero learnable parameters; BatchNorm's gamma/beta are
 * summed here directly since the module doesn't expose a helper). */
static int batchnorm_total_params(const BatchNorm *bn) {
    return (int)(bn->gamma->size + bn->beta->size);
}

void nn_print_summary(const Network *net) {
    printf("=== Network Summary ===\n");
    int total = 0;

    for (int i = 0; i < net->num_layers; i++) {
        const NetworkLayer *nl = &net->layers[i];
        int p = 0;

        switch (nl->type) {
        case LAYER_DENSE: {
            DenseLayer *l = (DenseLayer *)nl->layer_ptr;
            static const char *act_names[] = {"None","ReLU","Sigmoid","Tanh","Softmax"};
            p = dense_param_count(l);
            printf("  Layer %d: Dense(%d -> %d, %s)  params=%d\n",
                   i, l->in_features, l->out_features,
                   act_names[l->activation], p);
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayer *l = (DropoutLayer *)nl->layer_ptr;
            p = 0; /* no learnable parameters */
            printf("  Layer %d: Dropout(p=%.2f)  params=%d\n", i, l->drop_prob, p);
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNorm *l = (BatchNorm *)nl->layer_ptr;
            p = batchnorm_total_params(l);
            printf("  Layer %d: BatchNorm(%d, eps=%g, momentum=%.2f)  params=%d\n",
                   i, l->num_features, (double)l->eps, l->momentum, p);
            break;
        }
        case LAYER_RNN: {
            RNNLayer *l = (RNNLayer *)nl->layer_ptr;
            p = rnn_param_count(l);
            printf("  Layer %d: RNN(%d -> %d)  params=%d\n",
                   i, l->input_size, l->hidden_size, p);
            break;
        }
        case LAYER_LSTM: {
            LSTMLayer *l = (LSTMLayer *)nl->layer_ptr;
            p = lstm_param_count(l);
            printf("  Layer %d: LSTM(%d -> %d)  params=%d\n",
                   i, l->input_size, l->hidden_size, p);
            break;
        }
        default:
            printf("  Layer %d: <unrecognized type %d>\n", i, (int)nl->type);
            break;
        }
        total += p;
    }
    printf("  Total params: %d\n", total);
    printf("=======================\n");
}

int nn_total_params(const Network *net) {
    int t = 0;
    for (int i = 0; i < net->num_layers; i++) {
        const NetworkLayer *nl = &net->layers[i];
        switch (nl->type) {
        case LAYER_DENSE:     t += dense_param_count((DenseLayer *)nl->layer_ptr);        break;
        case LAYER_DROPOUT:   /* 0 params */                                              break;
        case LAYER_BATCHNORM: t += batchnorm_total_params((BatchNorm *)nl->layer_ptr);     break;
        case LAYER_RNN:       t += rnn_param_count((RNNLayer *)nl->layer_ptr);             break;
        case LAYER_LSTM:      t += lstm_param_count((LSTMLayer *)nl->layer_ptr);           break;
        default: break;
        }
    }
    return t;
}

/* ── metrics ─────────────────────────────────────────────────────── */

float nn_accuracy_binary(const Tensor *pred, const Tensor *target) {
    CF_CHECK(pred->size == target->size, "nn_accuracy_binary: size mismatch");
    int correct = 0;
    for (size_t i = 0; i < pred->size; i++) {
        int p = pred->data[i] >= 0.5f ? 1 : 0;
        int t = target->data[i] >= 0.5f ? 1 : 0;
        if (p == t) correct++;
    }
    return (float)correct / (float)pred->size;
}

float nn_accuracy_multiclass(const Tensor *pred, const Tensor *target) {
    CF_CHECK(pred->ndim == 2 && target->ndim == 2, "nn_accuracy_multiclass: expected 2D tensors");
    CF_CHECK(pred->shape[0] == target->shape[0] && pred->shape[1] == target->shape[1],
              "nn_accuracy_multiclass: shape mismatch");
    int batch   = pred->shape[0];
    int classes = pred->shape[1];
    int correct = 0;
    for (int b = 0; b < batch; b++) {
        /* argmax of pred row */
        int pred_cls = 0;
        float best   = pred->data[b * classes];
        for (int c = 1; c < classes; c++) {
            float v = pred->data[b * classes + c];
            if (v > best) { best = v; pred_cls = c; }
        }
        /* argmax of target row (handles one-hot) */
        int true_cls = 0;
        float tbest  = target->data[b * classes];
        for (int c = 1; c < classes; c++) {
            float v = target->data[b * classes + c];
            if (v > tbest) { tbest = v; true_cls = c; }
        }
        if (pred_cls == true_cls) correct++;
    }
    return (float)correct / (float)batch;
}

/* ══════════════════════════════════════════════════════════════════
 * UNIFIED MODEL SERIALIZATION  (see the format doc in nn.h)
 * ══════════════════════════════════════════════════════════════════ */

/* Generous but bounded sanity cap for any dimension decoded from a
 * file, applied *before* it's used to size a malloc/tensor_create —
 * this is what keeps a corrupted "in_features = 0x7fffffff" from
 * turning into a multi-gigabyte (or overflowing) allocation attempt. */
#define NN_MAX_REASONABLE_DIM (1 << 20)   /* 1,048,576 */

static int dim_ok(int32_t v) { return v > 0 && v <= NN_MAX_REASONABLE_DIM; }

int nn_save_model(Network *net, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "nn_save_model: cannot open '%s' for writing\n", path);
        return -1;
    }

    uint32_t magic      = NN_MODEL_MAGIC;
    int32_t  version    = NN_MODEL_VERSION;
    int32_t  num_layers = net->num_layers;

    if (fwrite(&magic,      sizeof(magic),      1, f) != 1 ||
        fwrite(&version,    sizeof(version),    1, f) != 1 ||
        fwrite(&num_layers, sizeof(num_layers), 1, f) != 1) {
        fprintf(stderr, "nn_save_model: write error (header)\n");
        fclose(f);
        return -1;
    }

    for (int i = 0; i < net->num_layers; i++) {
        NetworkLayer *nl       = &net->layers[i];
        int32_t       type_tag = (int32_t)nl->type;
        if (fwrite(&type_tag, sizeof(type_tag), 1, f) != 1) goto write_fail;

        switch (nl->type) {
        case LAYER_DENSE: {
            DenseLayer *l = (DenseLayer *)nl->layer_ptr;
            int32_t cfg[3] = { l->in_features, l->out_features, (int32_t)l->activation };
            if (fwrite(cfg, sizeof(int32_t), 3, f) != 3) goto write_fail;
            if (fwrite(l->W->data, sizeof(float), l->W->size, f) != l->W->size) goto write_fail;
            if (fwrite(l->b->data, sizeof(float), l->b->size, f) != l->b->size) goto write_fail;
            break;
        }
        case LAYER_DROPOUT: {
            DropoutLayer *l = (DropoutLayer *)nl->layer_ptr;
            float p = l->drop_prob;
            if (fwrite(&p, sizeof(p), 1, f) != 1) goto write_fail;
            /* no learnable parameters to persist */
            break;
        }
        case LAYER_BATCHNORM: {
            BatchNorm *l = (BatchNorm *)nl->layer_ptr;
            int32_t nf     = l->num_features;
            float   cfg[2] = { l->eps, l->momentum };
            if (fwrite(&nf, sizeof(nf),    1, f) != 1) goto write_fail;
            if (fwrite(cfg, sizeof(float), 2, f) != 2) goto write_fail;
            if (fwrite(l->gamma->data,        sizeof(float), l->gamma->size,        f) != l->gamma->size)        goto write_fail;
            if (fwrite(l->beta->data,         sizeof(float), l->beta->size,         f) != l->beta->size)         goto write_fail;
            if (fwrite(l->running_mean->data, sizeof(float), l->running_mean->size, f) != l->running_mean->size) goto write_fail;
            if (fwrite(l->running_var->data,  sizeof(float), l->running_var->size,  f) != l->running_var->size)  goto write_fail;
            break;
        }
        case LAYER_RNN: {
            RNNLayer *l = (RNNLayer *)nl->layer_ptr;
            int32_t cfg[2] = { l->input_size, l->hidden_size };
            if (fwrite(cfg, sizeof(int32_t), 2, f) != 2) goto write_fail;
            if (fwrite(l->W_xh->data, sizeof(float), l->W_xh->size, f) != l->W_xh->size) goto write_fail;
            if (fwrite(l->W_hh->data, sizeof(float), l->W_hh->size, f) != l->W_hh->size) goto write_fail;
            if (fwrite(l->b_h->data,  sizeof(float), l->b_h->size,  f) != l->b_h->size)  goto write_fail;
            break;
        }
        case LAYER_LSTM: {
            LSTMLayer *l = (LSTMLayer *)nl->layer_ptr;
            int32_t cfg[2] = { l->input_size, l->hidden_size };
            if (fwrite(cfg, sizeof(int32_t), 2, f) != 2) goto write_fail;
            if (fwrite(l->W_ih->data, sizeof(float), l->W_ih->size, f) != l->W_ih->size) goto write_fail;
            if (fwrite(l->W_hh->data, sizeof(float), l->W_hh->size, f) != l->W_hh->size) goto write_fail;
            if (fwrite(l->b->data,    sizeof(float), l->b->size,    f) != l->b->size)    goto write_fail;
            break;
        }
        default:
            fprintf(stderr, "nn_save_model: unrecognized LayerType %d at layer %d\n",
                    (int)nl->type, i);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;

write_fail:
    fprintf(stderr, "nn_save_model: write error (I/O failure or disk full)\n");
    fclose(f);
    return -1;
}

int nn_load_model(Network *net, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nn_load_model: cannot open '%s'\n", path);
        return -1;
    }

    uint32_t magic;
    int32_t  version, num_layers;

    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != NN_MODEL_MAGIC) {
        fprintf(stderr, "nn_load_model: bad magic number — not a libneuralc "
                        "model file, or the file is corrupted\n");
        fclose(f);
        return -1;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != NN_MODEL_VERSION) {
        fprintf(stderr, "nn_load_model: unsupported or corrupt file version\n");
        fclose(f);
        return -1;
    }
    if (fread(&num_layers, sizeof(num_layers), 1, f) != 1 ||
        num_layers < 0 || num_layers > NN_MAX_LAYERS) {
        fprintf(stderr, "nn_load_model: invalid layer count in file header\n");
        fclose(f);
        return -1;
    }

    /* Discard whatever `net` currently holds — nn_load_model rebuilds
     * the network from scratch via the factory switch below. */
    nn_clear_layers(net);

    for (int i = 0; i < num_layers; i++) {
        int32_t type_tag;
        if (fread(&type_tag, sizeof(type_tag), 1, f) != 1) goto read_fail;

        switch ((LayerType)type_tag) {
        case LAYER_DENSE: {
            int32_t cfg[3];
            if (fread(cfg, sizeof(int32_t), 3, f) != 3) goto read_fail;
            int32_t in_f = cfg[0], out_f = cfg[1], act = cfg[2];
            if (!dim_ok(in_f) || !dim_ok(out_f) || act < ACT_NONE || act > ACT_SOFTMAX) {
                fprintf(stderr, "nn_load_model: corrupt Dense config at layer %d\n", i);
                goto read_fail;
            }
            DenseLayer *l = dense_create(in_f, out_f, (Activation)act);
            if (!l) goto read_fail;
            if (fread(l->W->data, sizeof(float), l->W->size, f) != l->W->size ||
                fread(l->b->data, sizeof(float), l->b->size, f) != l->b->size) {
                dense_free(l);
                goto read_fail;
            }
            nn_add_dense(net, l);
            break;
        }
        case LAYER_DROPOUT: {
            float drop_prob;
            if (fread(&drop_prob, sizeof(drop_prob), 1, f) != 1) goto read_fail;
            if (!(drop_prob >= 0.0f && drop_prob < 1.0f)) {
                fprintf(stderr, "nn_load_model: corrupt Dropout config at layer %d\n", i);
                goto read_fail;
            }
            DropoutLayer *l = dropout_create(drop_prob);
            if (!l) goto read_fail;
            nn_add_dropout(net, l);
            break;
        }
        case LAYER_BATCHNORM: {
            int32_t nf;
            float   cfg[2];
            if (fread(&nf,  sizeof(nf),        1, f) != 1 ||
                fread(cfg,  sizeof(float),     2, f) != 2) goto read_fail;
            float eps = cfg[0], momentum = cfg[1];
            if (!dim_ok(nf) || eps <= 0.0f || momentum < 0.0f || momentum > 1.0f) {
                fprintf(stderr, "nn_load_model: corrupt BatchNorm config at layer %d\n", i);
                goto read_fail;
            }
            BatchNorm *l = batchnorm_create(nf, eps, momentum);
            if (!l) goto read_fail;
            if (fread(l->gamma->data,        sizeof(float), l->gamma->size,        f) != l->gamma->size        ||
                fread(l->beta->data,         sizeof(float), l->beta->size,         f) != l->beta->size         ||
                fread(l->running_mean->data, sizeof(float), l->running_mean->size, f) != l->running_mean->size ||
                fread(l->running_var->data,  sizeof(float), l->running_var->size,  f) != l->running_var->size) {
                batchnorm_free(l);
                goto read_fail;
            }
            nn_add_batchnorm(net, l);
            break;
        }
        case LAYER_RNN: {
            int32_t cfg[2];
            if (fread(cfg, sizeof(int32_t), 2, f) != 2) goto read_fail;
            int32_t in_f = cfg[0], hidden = cfg[1];
            if (!dim_ok(in_f) || !dim_ok(hidden)) {
                fprintf(stderr, "nn_load_model: corrupt RNN config at layer %d\n", i);
                goto read_fail;
            }
            RNNLayer *l = rnn_create(in_f, hidden);
            if (!l) goto read_fail;
            if (fread(l->W_xh->data, sizeof(float), l->W_xh->size, f) != l->W_xh->size ||
                fread(l->W_hh->data, sizeof(float), l->W_hh->size, f) != l->W_hh->size ||
                fread(l->b_h->data,  sizeof(float), l->b_h->size,  f) != l->b_h->size) {
                rnn_free(l);
                goto read_fail;
            }
            nn_add_rnn(net, l);
            break;
        }
        case LAYER_LSTM: {
            int32_t cfg[2];
            if (fread(cfg, sizeof(int32_t), 2, f) != 2) goto read_fail;
            int32_t in_f = cfg[0], hidden = cfg[1];
            if (!dim_ok(in_f) || !dim_ok(hidden)) {
                fprintf(stderr, "nn_load_model: corrupt LSTM config at layer %d\n", i);
                goto read_fail;
            }
            LSTMLayer *l = lstm_create(in_f, hidden);
            if (!l) goto read_fail;
            if (fread(l->W_ih->data, sizeof(float), l->W_ih->size, f) != l->W_ih->size ||
                fread(l->W_hh->data, sizeof(float), l->W_hh->size, f) != l->W_hh->size ||
                fread(l->b->data,    sizeof(float), l->b->size,    f) != l->b->size) {
                lstm_free(l);
                goto read_fail;
            }
            nn_add_lstm(net, l);
            break;
        }
        default:
            fprintf(stderr, "nn_load_model: unrecognized LayerType %d at layer %d "
                            "(corrupt file?)\n", (int)type_tag, i);
            goto read_fail;
        }
    }

    fclose(f);
    return 0;

read_fail:
    fprintf(stderr, "nn_load_model: failed while reading layer data — file is "
                    "truncated or corrupt; discarding partially-loaded model\n");
    fclose(f);
    nn_clear_layers(net);   /* unwind whatever we'd already constructed */
    return -1;
}

/* ── legacy weight I/O ─────────────────────────────────────────────
 * Kept for existing call sites; both now just forward to the unified
 * model (de)serializer, which subsumes their old Dense-only behavior
 * (and, as a bonus, now also round-trips heterogeneous networks). */

int nn_save(const Network *net, const char *path) {
    /* nn_save_model() never mutates *net despite the non-const
     * signature — it only reads layer state to write it out — so
     * this cast is safe. */
    return nn_save_model((Network *)net, path);
}

int nn_load(Network *net, const char *path) {
    return nn_load_model(net, path);
}
