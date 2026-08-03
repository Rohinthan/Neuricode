/*
 * model.c — reusable inference-time Model abstraction.
 *
 * Everything CLI-specific (argv parsing, usage text, stdout streaming,
 * srand/time-seeding, neuralc_memory_init/reset) has moved to cli.c.
 * This file knows nothing about argv or stdio; on failure it writes a
 * message into a caller-supplied buffer instead of printing directly,
 * so it's equally usable from a server, a test harness, or a CLI.
 *
 * ── Supported architecture ─────────────────────────────────────────
 * Exactly two layers, in this order:
 *   [0] LAYER_RNN or LAYER_LSTM   — recurrent encoder
 *   [1] LAYER_DENSE (ACT_SOFTMAX) — output projection to vocabulary
 *
 * Same reasoning as before: nn_forward()/nn_backward() don't chain
 * RNN/LSTM into Dense (nn.h documents this directly), and Dense's
 * in-Network forward always feeds it a *fresh* zero initial state —
 * there's nowhere to persist hidden/cell state across nn_forward()
 * calls. So this file drives rnn_forward()/lstm_forward() and
 * dense_forward() directly, one timestep at a time, and owns the
 * carried Tensor* hidden/cell state itself.
 *
 * Vocab size is no longer a compile-time #define — it's read off the
 * loaded Dense layer's out_features at model_load() time and stored
 * on the Model, so a differently-sized vocabulary (e.g. a future
 * non-byte-level tokenizer) works without touching this file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "tensor.h"
#include "layer.h"
#include "rnn.h"
#include "nn.h"
#include "model.h"

/* ═══════════════════════════════════════════════════════════════════
 *  Recurrent-layer abstraction — hides the RNN/LSTM split behind one
 *  call. Hidden/cell state is Tensor* (the library's real API), NULL
 *  meaning "zero-initialize" per rnn_forward()/lstm_forward().
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    LayerType  type;         /* LAYER_RNN or LAYER_LSTM only */
    RNNLayer  *rnn;          /* valid iff type == LAYER_RNN  */
    LSTMLayer *lstm;         /* valid iff type == LAYER_LSTM */
    Tensor    *h_prev;       /* carried hidden state; NULL = zero init */
    Tensor    *c_prev;       /* carried cell state (LSTM only)         */
    int        input_size;
    int        hidden_size;
} Recurrent;

static void recurrent_reset(Recurrent *r) {
    r->h_prev = NULL;
    r->c_prev = NULL;
}

/* One timestep: x is [1,1,input_size], out is [1,1,hidden_size]
 * (both allocated once by the caller and reused every call). Updates
 * h_prev/c_prev in place for the next call. */
static void recurrent_step(Recurrent *r, const Tensor *x, Tensor *out) {
    if (r->type == LAYER_RNN) {
        rnn_forward(r->rnn, x, r->h_prev, out);
        r->h_prev = r->rnn->h_states[1];
    } else {
        lstm_forward(r->lstm, x, r->h_prev, r->c_prev, out);
        r->h_prev = r->lstm->h_state[1];
        r->c_prev = r->lstm->c_state[1];
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  Model — the opaque struct declared in model.h
 * ═══════════════════════════════════════════════════════════════════ */

struct Model {
    Network    *net;          /* owned; freed by model_free */
    Recurrent   rec;
    DenseLayer *dense;         /* borrowed from net — not separately owned */
    int         vocab_size;
    int         hidden_size;

    /* scratch buffers, allocated once at load time, reused forever */
    Tensor     *x;             /* [1,1,vocab_size]  one-hot input       */
    Tensor     *rec_out;       /* [1,1,hidden_size] recurrent output    */
    Tensor     *rec_out_2d;    /* view of rec_out reshaped to 2D, for   *
                                 * dense_forward — see the note on this  *
                                 * in the original file: rec_out's data  *
                                 * buffer is caller-owned and reused in  *
                                 * place by every rnn/lstm_forward call, *
                                 * so this reshape is a metadata view    *
                                 * created once, not once per step.      */
    Tensor     *dense_out;     /* [1,vocab_size]    softmax distribution*/
};

/* ═══════════════════════════════════════════════════════════════════
 *  Error reporting helper — snprintf into caller's buffer, safe under
 *  err == NULL / err_len == 0 (both model_load()'s contract).
 * ═══════════════════════════════════════════════════════════════════ */

static void set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Architecture validation — same rules as the original CLI's
 *  extract_layers(), just reporting through set_err() instead of
 *  fprintf(stderr, ...).
 * ═══════════════════════════════════════════════════════════════════ */

static int extract_layers(Network *net, Recurrent *rec, DenseLayer **dense,
                           char *err, size_t err_len) {
    if (net->num_layers != 2) {
        set_err(err, err_len,
            "invalid architecture: expected exactly 2 layers "
            "(recurrent + dense), got %d", net->num_layers);
        return -1;
    }

    NetworkLayer *l0 = &net->layers[0];
    NetworkLayer *l1 = &net->layers[1];

    if (l0->type == LAYER_RNN) {
        if (!l0->layer_ptr) {
            set_err(err, err_len, "invalid model file: layer 0 has a null pointer");
            return -1;
        }
        rec->type        = LAYER_RNN;
        rec->rnn         = (RNNLayer *)l0->layer_ptr;
        rec->lstm        = NULL;
        rec->input_size  = rec->rnn->input_size;
        rec->hidden_size = rec->rnn->hidden_size;
    } else if (l0->type == LAYER_LSTM) {
        if (!l0->layer_ptr) {
            set_err(err, err_len, "invalid model file: layer 0 has a null pointer");
            return -1;
        }
        rec->type        = LAYER_LSTM;
        rec->lstm        = (LSTMLayer *)l0->layer_ptr;
        rec->rnn         = NULL;
        rec->input_size  = rec->lstm->input_size;
        rec->hidden_size = rec->lstm->hidden_size;
    } else {
        set_err(err, err_len, "invalid architecture: layer 0 must be RNN or LSTM");
        return -1;
    }

    if (l1->type != LAYER_DENSE) {
        set_err(err, err_len, "invalid architecture: layer 1 must be Dense");
        return -1;
    }
    if (!l1->layer_ptr) {
        set_err(err, err_len, "invalid model file: layer 1 has a null pointer");
        return -1;
    }
    *dense = (DenseLayer *)l1->layer_ptr;

    if ((*dense)->activation != ACT_SOFTMAX) {
        set_err(err, err_len,
            "invalid architecture: Dense layer must use softmax activation");
        return -1;
    }

    if ((*dense)->in_features != rec->hidden_size) {
        set_err(err, err_len,
            "dimension mismatch: dense in_features (%d) != "
            "recurrent hidden_size (%d)",
            (*dense)->in_features, rec->hidden_size);
        return -1;
    }

    if ((*dense)->out_features != rec->input_size) {
        set_err(err, err_len,
            "dimension mismatch: dense out_features (%d) != "
            "recurrent input_size (%d) — vocab size must agree "
            "on both ends of the loop",
            (*dense)->out_features, rec->input_size);
        return -1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  model_load / model_free
 * ═══════════════════════════════════════════════════════════════════ */

Model *model_load(const char *path, char *err, size_t err_len) {
    if (!path) {
        set_err(err, err_len, "null path passed to model_load");
        return NULL;
    }

    Network *net = nn_create();
    if (!net) {
        set_err(err, err_len, "failed to allocate network");
        return NULL;
    }
    if (nn_load_model(net, path) != 0) {
        set_err(err, err_len, "failed to load model from '%s'", path);
        nn_free(net);
        return NULL;
    }

    Model *m = (Model *)calloc(1, sizeof(Model));
    if (!m) {
        set_err(err, err_len, "failed to allocate Model struct");
        nn_free(net);
        return NULL;
    }
    m->net = net;

    if (extract_layers(net, &m->rec, &m->dense, err, err_len) != 0) {
        nn_free(net);
        free(m);
        return NULL;
    }

    m->vocab_size  = m->rec.input_size;
    m->hidden_size = m->rec.hidden_size;

    int sh_in[3]        = {1, 1, m->vocab_size};
    int sh_rec_out[3]   = {1, 1, m->hidden_size};
    int sh_dense_in[2]  = {1, m->hidden_size};
    int sh_dense_out[2] = {1, m->vocab_size};

    m->x         = tensor_zeros(sh_in, 3);
    m->rec_out   = tensor_zeros(sh_rec_out, 3);
    m->dense_out = tensor_zeros(sh_dense_out, 2);

    if (!m->x || !m->rec_out || !m->dense_out) {
        set_err(err, err_len, "failed to allocate inference buffers");
        model_free(m);
        return NULL;
    }

    /* Safe to create once here and reuse for every step — see the
     * struct-field comment on rec_out_2d above for why this is only a
     * metadata view rather than a per-step allocation. */
    m->rec_out_2d = tensor_reshape(m->rec_out, sh_dense_in, 2);
    if (!m->rec_out_2d) {
        set_err(err, err_len, "failed to reshape recurrent output for Dense input");
        model_free(m);
        return NULL;
    }

    recurrent_reset(&m->rec);
    return m;
}

void model_free(Model *m) {
    if (!m) return;
    tensor_free(m->rec_out_2d);
    tensor_free(m->x);
    tensor_free(m->rec_out);
    tensor_free(m->dense_out);
    nn_free(m->net);
    free(m);
}

void model_reset_state(Model *m) {
    if (!m) return;
    recurrent_reset(&m->rec);
}

int model_vocab_size(const Model *m)  { return m ? m->vocab_size  : 0; }
int model_hidden_size(const Model *m) { return m ? m->hidden_size : 0; }

/* ═══════════════════════════════════════════════════════════════════
 *  model_step
 * ═══════════════════════════════════════════════════════════════════ */

int model_step(Model *m, int token_id, float *out_probs) {
    if (!m || !out_probs) return -1;
    if (token_id < 0 || token_id >= m->vocab_size) return -1;

    tensor_fill(m->x, 0.0f);
    m->x->data[token_id] = 1.0f;

    recurrent_step(&m->rec, m->x, m->rec_out);
    dense_forward(m->dense, m->rec_out_2d, m->dense_out);

    memcpy(out_probs, m->dense_out->data, (size_t)m->vocab_size * sizeof(float));
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  model_generate
 * ═══════════════════════════════════════════════════════════════════ */

int model_generate(Model *m, const unsigned char *prompt, size_t prompt_len,
                    int max_tokens, int eos_token, const Sampler *sampler,
                    void (*cb)(int token, void *user_data), void *user_data) {
    if (!m || !sampler) return -1;
    if (prompt_len == 0) return -1; /* no BOS/seed-token in this byte-level
                                      * architecture — at least one prompt
                                      * byte is required to produce a first
                                      * distribution to sample from */

    for (size_t i = 0; i < prompt_len; i++) {
        if ((int)prompt[i] >= m->vocab_size) return -1;
    }

    /* sampler_pick mutates rng_state, so take a local mutable copy —
     * model_generate's own Sampler parameter stays const/untouched,
     * matching the "sampler describes a strategy, doesn't own a
     * conversation's worth of state" mental model. Callers who want
     * the stream to persist across multiple model_generate() calls
     * should keep and pass their own Sampler variable instead. */
    Sampler local = *sampler;

    float *logits = (float *)malloc((size_t)m->vocab_size * sizeof(float));
    float *probs  = (float *)malloc((size_t)m->vocab_size * sizeof(float));
    float *step_probs = (float *)malloc((size_t)m->vocab_size * sizeof(float));
    if (!logits || !probs || !step_probs) {
        free(logits); free(probs); free(step_probs);
        return -1;
    }

    /* ── prompt warmup: no callback invoked during this phase ─────── */
    for (size_t i = 0; i < prompt_len; i++) {
        model_step(m, (int)prompt[i], step_probs);
    }

    /* dense_out already holds the distribution for the token *after*
     * everything fed so far — sample it before running the model
     * again, mirroring the original loop's ordering exactly. */
    for (int step = 0; step < max_tokens; step++) {
        int token = sampler_pick(&local, step_probs, m->vocab_size, logits, probs);
        if (token < 0) { free(logits); free(probs); free(step_probs); return -1; }

        cb(token, user_data);

        if (token == eos_token) break;
        if (step == max_tokens - 1) break; /* no need to run the model once more */

        model_step(m, token, step_probs);
    }

    free(logits);
    free(probs);
    free(step_probs);
    return 0;
}
