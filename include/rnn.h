#ifndef RNN_H
#define RNN_H

#include "tensor.h"

/*
 * rnn.h — Vanilla RNN and LSTM layers for neuralc
 *
 * Both layers process sequences of shape [batch, seq_len, input_size]
 * and produce outputs of shape [batch, seq_len, hidden_size].
 *
 * ── Vanilla RNN ──────────────────────────────────────────────────
 *   h[t] = tanh( x[t] @ W_xh^T + h[t-1] @ W_hh^T + b_h )
 *
 * ── LSTM ─────────────────────────────────────────────────────────
 *   gates       = x[t] @ W_ih^T + h[t-1] @ W_hh^T + b
 *   i[t]        = sigmoid(gates[0:H])    input gate
 *   f[t]        = sigmoid(gates[H:2H])   forget gate
 *   o[t]        = sigmoid(gates[2H:3H])  output gate
 *   g[t]        = tanh   (gates[3H:4H])  cell gate
 *   c[t]        = f[t]*c[t-1] + i[t]*g[t]
 *   h[t]        = o[t]*tanh(c[t])
 *
 * Training uses Backpropagation Through Time (BPTT).
 */

#define RNN_MAX_SEQ 512   /* max sequence length supported */

/* ── Vanilla RNN ─────────────────────────────────────────────────── */

typedef struct {
    int input_size;
    int hidden_size;

    /* parameters */
    Tensor *W_xh;   /* [hidden, input]   input → hidden  */
    Tensor *W_hh;   /* [hidden, hidden]  hidden → hidden */
    Tensor *b_h;    /* [hidden]          bias             */

    /* gradients */
    Tensor *dW_xh;
    Tensor *dW_hh;
    Tensor *db_h;
    Tensor *dX;     /* [batch, seq, input] grad w.r.t. input */

    /* BPTT cache (allocated on first forward) */
    Tensor *h_states[RNN_MAX_SEQ + 1]; /* h[0]=init, h[1..T]=outputs */
    Tensor *z_cache [RNN_MAX_SEQ];     /* pre-tanh values             */
    Tensor *x_cache [RNN_MAX_SEQ];     /* input at each timestep      */
    int     seq_cache;
    int     batch_cache;
} RNNLayer;

/* ── LSTM ────────────────────────────────────────────────────────── */

typedef struct {
    int input_size;
    int hidden_size;

    /*
     * Combined gate weights (input + hidden in one matrix each):
     *   W_ih [4*hidden, input]   — maps input  to all 4 gates
     *   W_hh [4*hidden, hidden]  — maps hidden to all 4 gates
     *   b    [4*hidden]          — bias for all 4 gates
     *
     * Gate ordering: [i | f | o | g]
     */
    Tensor *W_ih;
    Tensor *W_hh;
    Tensor *b;

    /* gradients */
    Tensor *dW_ih;
    Tensor *dW_hh;
    Tensor *db;
    Tensor *dX;     /* [batch, seq, input] */

    /* BPTT cache */
    Tensor *i_gate[RNN_MAX_SEQ];   /* input  gate  [batch, hidden] */
    Tensor *f_gate[RNN_MAX_SEQ];   /* forget gate  [batch, hidden] */
    Tensor *o_gate[RNN_MAX_SEQ];   /* output gate  [batch, hidden] */
    Tensor *g_gate[RNN_MAX_SEQ];   /* cell   gate  [batch, hidden] */
    Tensor *c_state[RNN_MAX_SEQ+1];/* cell   state [batch, hidden] */
    Tensor *h_state[RNN_MAX_SEQ+1];/* hidden state [batch, hidden] */
    Tensor *x_cache[RNN_MAX_SEQ];  /* input cache  [batch, input]  */
    int     seq_cache;
    int     batch_cache;
} LSTMLayer;

/* ═══════════════════════════════════════════════════════════════════
 *  Vanilla RNN API
 * ═══════════════════════════════════════════════════════════════════ */

/* Create RNN layer. Weights initialized with Xavier. */
RNNLayer *rnn_create(int input_size, int hidden_size);
void      rnn_free(RNNLayer *l);

/*
 * rnn_forward:
 *   input    [batch, seq_len, input_size]   — flat row-major
 *   h_init   [batch, hidden_size]           — initial hidden state
 *                                             (NULL = use zeros)
 *   output   [batch, seq_len, hidden_size]  — all timestep outputs
 *                                             (caller allocates)
 */
void rnn_forward(RNNLayer *l, const Tensor *input,
                 const Tensor *h_init, Tensor *output);

/*
 * rnn_backward:
 *   grad_out [batch, seq_len, hidden_size]  — upstream gradient
 *   Fills l->dW_xh, l->dW_hh, l->db_h, l->dX after call.
 */
void rnn_backward(RNNLayer *l, const Tensor *grad_out);

/* Zero all gradients */
void rnn_zero_grad(RNNLayer *l);

/* SGD update (call after rnn_backward) */
void rnn_update_sgd(RNNLayer *l, float lr);

int rnn_param_count(const RNNLayer *l);

/* ═══════════════════════════════════════════════════════════════════
 *  LSTM API
 * ═══════════════════════════════════════════════════════════════════ */

/* Create LSTM layer. Weights initialized with Xavier. */
LSTMLayer *lstm_create(int input_size, int hidden_size);
void       lstm_free(LSTMLayer *l);

/*
 * lstm_forward:
 *   input    [batch, seq_len, input_size]
 *   h_init   [batch, hidden_size]  initial hidden (NULL = zeros)
 *   c_init   [batch, hidden_size]  initial cell   (NULL = zeros)
 *   output   [batch, seq_len, hidden_size]  (caller allocates)
 */
void lstm_forward(LSTMLayer *l, const Tensor *input,
                  const Tensor *h_init, const Tensor *c_init,
                  Tensor *output);

/*
 * lstm_backward:
 *   grad_out [batch, seq_len, hidden_size]
 *   Fills l->dW_ih, l->dW_hh, l->db, l->dX.
 */
void lstm_backward(LSTMLayer *l, const Tensor *grad_out);

void lstm_zero_grad(LSTMLayer *l);
void lstm_update_sgd(LSTMLayer *l, float lr);
int  lstm_param_count(const LSTMLayer *l);

/* ── gradient clipping ──────────────────────────────────────────── */
void rnn_clip_gradients(RNNLayer *l, float max_norm);
void lstm_clip_gradients(LSTMLayer *l, float max_norm);

#endif /* RNN_H */
