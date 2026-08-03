#include "rnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── allocation helpers ──────────────────────────────────────────── */

static Tensor *alloc2(int r, int c) {
    int sh[2] = {r, c}; return tensor_zeros(sh, 2);
}
static Tensor *alloc1(int n) {
    int sh[1] = {n}; return tensor_zeros(sh, 1);
}

/* Xavier init: scale = sqrt(1/fan_in) */
static void xavier_init(Tensor *W, int fan_in) {
    float scale = sqrtf(1.0f / (float)fan_in);
    int sh[2] = {W->shape[0], W->shape[1]};
    Tensor *tmp = tensor_randn(sh, 2);
    tensor_scale(tmp, scale, W);
    tensor_free(tmp);
}

/* matmul helper: out = A @ B^T  [M,K] @ [N,K]^T → [M,N] */
static void matmul_bt(const Tensor *A, const Tensor *B, Tensor *out) {
    /* A [M,K]  B [N,K]  out [M,N] */
    int M = A->shape[0], K = A->shape[1], N = B->shape[0];
    tensor_fill(out, 0.0f);
    // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
    // Each m owns out->data[m*N .. m*N+N-1] exclusively — the k,n
    // loops below never touch another row, so splitting the outer
    // loop across threads is safe: no cross-thread dependency.
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float a = A->data[m*K + k];
            for (int n = 0; n < N; n++)
                out->data[m*N + n] += a * B->data[n*K + k];
        }
}

/* ═══════════════════════════════════════════════════════════════════
 *  VANILLA RNN
 * ═══════════════════════════════════════════════════════════════════ */

RNNLayer *rnn_create(int input_size, int hidden_size) {
    RNNLayer *l = (RNNLayer *)calloc(1, sizeof(RNNLayer));
    CF_CHECK_ALLOC(l);
    l->input_size  = input_size;
    l->hidden_size = hidden_size;

    l->W_xh  = alloc2(hidden_size, input_size);
    l->W_hh  = alloc2(hidden_size, hidden_size);
    l->b_h   = alloc1(hidden_size);
    l->dW_xh = alloc2(hidden_size, input_size);
    l->dW_hh = alloc2(hidden_size, hidden_size);
    l->db_h  = alloc1(hidden_size);

    xavier_init(l->W_xh, input_size);
    xavier_init(l->W_hh, hidden_size);
    tensor_fill(l->b_h, 0.0f);
    return l;
}

void rnn_free(RNNLayer *l) {
    if (!l) return;
    tensor_free(l->W_xh); tensor_free(l->W_hh); tensor_free(l->b_h);
    tensor_free(l->dW_xh);tensor_free(l->dW_hh);tensor_free(l->db_h);
    tensor_free(l->dX);
    for (int t = 0; t <= l->seq_cache; t++) tensor_free(l->h_states[t]);
    for (int t = 0; t <  l->seq_cache; t++) {
        tensor_free(l->z_cache[t]);
        tensor_free(l->x_cache[t]);
    }
    free(l);
}

/* ── RNN forward ─────────────────────────────────────────────────── */

void rnn_forward(RNNLayer *l, const Tensor *input,
                 const Tensor *h_init, Tensor *output) {
    /*
     * input  [batch, seq, input_size]
     * output [batch, seq, hidden_size]
     */
    int batch = input->shape[0];
    int seq   = input->shape[1];
    int I     = l->input_size;
    int H     = l->hidden_size;

    CF_CHECK(seq <= RNN_MAX_SEQ, "rnn_forward: seq_len exceeds RNN_MAX_SEQ");
    CF_CHECK(input->shape[2] == I,
             "rnn_forward: input_size mismatch");

    /* free old cache if seq/batch changed */
    if (l->seq_cache != seq || l->batch_cache != batch) {
        for (int t = 0; t <= l->seq_cache; t++) {
            tensor_free(l->h_states[t]); l->h_states[t] = NULL;
        }
        for (int t = 0; t < l->seq_cache; t++) {
            tensor_free(l->z_cache[t]);  l->z_cache[t]  = NULL;
            tensor_free(l->x_cache[t]);  l->x_cache[t]  = NULL;
        }
        l->seq_cache   = seq;
        l->batch_cache = batch;
        for (int t = 0; t <= seq; t++) l->h_states[t] = alloc2(batch, H);
        for (int t = 0; t < seq;  t++) {
            l->z_cache[t] = alloc2(batch, H);
            l->x_cache[t] = alloc2(batch, I);
        }
    }

    /* h[0] = h_init or zeros */
    if (h_init)
        tensor_copy_data(l->h_states[0], h_init);
    else
        tensor_fill(l->h_states[0], 0.0f);

    Tensor *xh_out = alloc2(batch, H);
    Tensor *hh_out = alloc2(batch, H);

    // KEEP SEQUENTIAL: h_states[t+1] depends on h_states[t], so this
    // time loop must not be parallelized — only the per-batch work
    // inside each timestep is a candidate (marked below).
    for (int t = 0; t < seq; t++) {
        // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
        // Each b writes only its own row of x_cache[t] — independent.
        /* extract x[t] from input: rows [batch, I] */
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < I; i++)
                l->x_cache[t]->data[b*I + i] =
                    input->data[b*(seq*I) + t*I + i];

        /* z[t] = x[t] @ W_xh^T + h[t-1] @ W_hh^T + b_h */
        matmul_bt(l->x_cache[t], l->W_xh, xh_out);
        matmul_bt(l->h_states[t], l->W_hh, hh_out);

        // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
        // Each (b,h) pair reads only xh_out/hh_out at its own offset
        // and writes only its own z_cache[t]/h_states[t+1] entry —
        // no cross-thread dependency, safe to split on b.
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++) {
                float z = xh_out->data[b*H+h]
                        + hh_out->data[b*H+h]
                        + l->b_h->data[h];
                l->z_cache[t]->data[b*H+h]   = z;
                l->h_states[t+1]->data[b*H+h] = tanhf(z);
            }

        // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
        // Each b writes only its own row of output — independent.
        /* write h[t+1] to output */
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++)
                output->data[b*(seq*H) + t*H + h] =
                    l->h_states[t+1]->data[b*H+h];
    }

    tensor_free(xh_out);
    tensor_free(hh_out);
}

/* ── RNN backward (BPTT) ─────────────────────────────────────────── */

void rnn_backward(RNNLayer *l, const Tensor *grad_out) {
    int batch = l->batch_cache;
    int seq   = l->seq_cache;
    int I     = l->input_size;
    int H     = l->hidden_size;

    rnn_zero_grad(l);

    /* dX allocation */
    tensor_free(l->dX);
    int sh3[3] = {batch, seq, I};
    l->dX = tensor_zeros(sh3, 3);

    Tensor *dh_next = alloc2(batch, H);   /* dL/dh from next timestep */
    Tensor *dz      = alloc2(batch, H);   /* dL/dz (pre-tanh)         */
    Tensor *dx_t    = alloc2(batch, I);   /* dL/dx at timestep t      */

    tensor_fill(dh_next, 0.0f);

    /* BPTT: iterate backwards */
    for (int t = seq - 1; t >= 0; t--) {
        /* dh[t] = grad_out[t] + dh_next */
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++) {
                float dh = grad_out->data[b*(seq*H) + t*H + h]
                         + dh_next->data[b*H + h];
                /* tanh gradient: dz = dh * (1 - tanh^2(z)) */
                float th = l->h_states[t+1]->data[b*H + h];
                dz->data[b*H + h] = dh * (1.0f - th*th);
            }

        /* dW_xh += dz^T @ x[t]  accumulated */
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++) {
                float d = dz->data[b*H + h];
                for (int i = 0; i < I; i++)
                    l->dW_xh->data[h*I + i] +=
                        d * l->x_cache[t]->data[b*I + i];
            }

        /* dW_hh += h[t]^T @ dz */
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++) {
                float d = dz->data[b*H + h];
                for (int hh = 0; hh < H; hh++)
                    l->dW_hh->data[h*H + hh] +=
                        d * l->h_states[t]->data[b*H + hh];
            }

        /* db_h += sum over batch of dz */
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++)
                l->db_h->data[h] += dz->data[b*H + h];

        /* dx[t] = dz @ W_xh  [batch,H] @ [H,I] → [batch,I] */
        tensor_fill(dx_t, 0.0f);
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++) {
                float d = dz->data[b*H + h];
                for (int i = 0; i < I; i++)
                    dx_t->data[b*I + i] += d * l->W_xh->data[h*I + i];
            }

        /* store in dX */
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < I; i++)
                l->dX->data[b*(seq*I) + t*I + i] = dx_t->data[b*I + i];

        /* dh_next = dz @ W_hh  [batch,H] @ [H,H] → [batch,H] */
        tensor_fill(dh_next, 0.0f);
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < H; h++) {
                float d = dz->data[b*H + h];
                for (int hh = 0; hh < H; hh++)
                    dh_next->data[b*H + hh] += d * l->W_hh->data[h*H + hh];
            }
    }

    /* average gradients over batch and seq */
    float scale = 1.0f / (float)(batch * seq);
    tensor_scale(l->dW_xh, scale, l->dW_xh);
    tensor_scale(l->dW_hh, scale, l->dW_hh);
    tensor_scale(l->db_h,  scale, l->db_h);

    tensor_free(dh_next);
    tensor_free(dz);
    tensor_free(dx_t);
}

void rnn_zero_grad(RNNLayer *l) {
    tensor_fill(l->dW_xh, 0.0f);
    tensor_fill(l->dW_hh, 0.0f);
    tensor_fill(l->db_h,  0.0f);
}

void rnn_update_sgd(RNNLayer *l, float lr) {
    for (size_t i = 0; i < l->W_xh->size; i++)
        l->W_xh->data[i] -= lr * l->dW_xh->data[i];
    for (size_t i = 0; i < l->W_hh->size; i++)
        l->W_hh->data[i] -= lr * l->dW_hh->data[i];
    for (size_t i = 0; i < l->b_h->size; i++)
        l->b_h->data[i]  -= lr * l->db_h->data[i];
}

int rnn_param_count(const RNNLayer *l) {
    return (int)(l->W_xh->size + l->W_hh->size + l->b_h->size);
}

/* ═══════════════════════════════════════════════════════════════════
 *  LSTM
 * ═══════════════════════════════════════════════════════════════════ */

LSTMLayer *lstm_create(int input_size, int hidden_size) {
    LSTMLayer *l = (LSTMLayer *)calloc(1, sizeof(LSTMLayer));
    CF_CHECK_ALLOC(l);
    l->input_size  = input_size;
    l->hidden_size = hidden_size;

    int H4 = 4 * hidden_size;
    l->W_ih  = alloc2(H4, input_size);
    l->W_hh  = alloc2(H4, hidden_size);
    l->b     = alloc1(H4);
    l->dW_ih = alloc2(H4, input_size);
    l->dW_hh = alloc2(H4, hidden_size);
    l->db    = alloc1(H4);

    xavier_init(l->W_ih, input_size);
    xavier_init(l->W_hh, hidden_size);

    /* init forget gate bias to 1 — helps vanishing gradient */
    for (int h = hidden_size; h < 2*hidden_size; h++)
        l->b->data[h] = 1.0f;

    return l;
}

void lstm_free(LSTMLayer *l) {
    if (!l) return;
    tensor_free(l->W_ih); tensor_free(l->W_hh); tensor_free(l->b);
    tensor_free(l->dW_ih);tensor_free(l->dW_hh);tensor_free(l->db);
    tensor_free(l->dX);
    for (int t = 0; t <= l->seq_cache; t++) {
        tensor_free(l->c_state[t]);
        tensor_free(l->h_state[t]);
    }
    for (int t = 0; t < l->seq_cache; t++) {
        tensor_free(l->i_gate[t]); tensor_free(l->f_gate[t]);
        tensor_free(l->o_gate[t]); tensor_free(l->g_gate[t]);
        tensor_free(l->x_cache[t]);
    }
    free(l);
}

/* ── LSTM forward ────────────────────────────────────────────────── */

void lstm_forward(LSTMLayer *l, const Tensor *input,
                  const Tensor *h_init, const Tensor *c_init,
                  Tensor *output) {
    int batch = input->shape[0];
    int seq   = input->shape[1];
    int I     = l->input_size;
    int H     = l->hidden_size;
    int H4    = 4 * H;

    CF_CHECK(seq <= RNN_MAX_SEQ, "lstm_forward: seq_len exceeds RNN_MAX_SEQ");
    CF_CHECK(input->shape[2] == I, "lstm_forward: input_size mismatch");

    /* reallocate cache if needed */
    if (l->seq_cache != seq || l->batch_cache != batch) {
        for (int t = 0; t <= l->seq_cache; t++) {
            tensor_free(l->c_state[t]); l->c_state[t] = NULL;
            tensor_free(l->h_state[t]); l->h_state[t] = NULL;
        }
        for (int t = 0; t < l->seq_cache; t++) {
            tensor_free(l->i_gate[t]); l->i_gate[t] = NULL;
            tensor_free(l->f_gate[t]); l->f_gate[t] = NULL;
            tensor_free(l->o_gate[t]); l->o_gate[t] = NULL;
            tensor_free(l->g_gate[t]); l->g_gate[t] = NULL;
            tensor_free(l->x_cache[t]);l->x_cache[t] = NULL;
        }
        l->seq_cache   = seq;
        l->batch_cache = batch;
        for (int t = 0; t <= seq; t++) {
            l->c_state[t] = alloc2(batch, H);
            l->h_state[t] = alloc2(batch, H);
        }
        for (int t = 0; t < seq; t++) {
            l->i_gate[t] = alloc2(batch, H);
            l->f_gate[t] = alloc2(batch, H);
            l->o_gate[t] = alloc2(batch, H);
            l->g_gate[t] = alloc2(batch, H);
            l->x_cache[t]= alloc2(batch, I);
        }
    }

    /* initial states */
    if (h_init) tensor_copy_data(l->h_state[0], h_init);
    else        tensor_fill(l->h_state[0], 0.0f);
    if (c_init) tensor_copy_data(l->c_state[0], c_init);
    else        tensor_fill(l->c_state[0], 0.0f);

    Tensor *gates_raw = alloc2(batch, H4);
    Tensor *xh_out   = alloc2(batch, H4);
    Tensor *hh_out   = alloc2(batch, H4);

    // KEEP SEQUENTIAL: c_state[t+1]/h_state[t+1] depend on
    // c_state[t]/h_state[t], so this time loop must not be
    // parallelized — only the per-batch work inside each timestep
    // is a candidate (marked below).
    for (int t = 0; t < seq; t++) {
        // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
        // Each b writes only its own row of x_cache[t] — independent.
        /* cache x[t] */
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < I; i++)
                l->x_cache[t]->data[b*I+i] =
                    input->data[b*(seq*I) + t*I + i];

        /* gates_raw = x[t] @ W_ih^T + h[t-1] @ W_hh^T + b */
        matmul_bt(l->x_cache[t],  l->W_ih, xh_out);
        matmul_bt(l->h_state[t],  l->W_hh, hh_out);

        // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
        // Each (b,g) pair writes only its own gates_raw entry.
        for (int b = 0; b < batch; b++)
            for (int g = 0; g < H4; g++)
                gates_raw->data[b*H4+g] = xh_out->data[b*H4+g]
                                        + hh_out->data[b*H4+g]
                                        + l->b->data[g];

        // TODO: PARALLELIZE_OUTER_LOOP (OpenMP ready)
        // Each b reads only its own slice of gates_raw and writes
        // only its own row of i/f/o/g_gate, c_state[t+1], h_state[t+1],
        // and output — no cross-thread dependency, safe to split on b.
        /* split and activate gates */
        for (int b = 0; b < batch; b++) {
            float *gr = gates_raw->data + b*H4;
            for (int h = 0; h < H; h++) {
                float i_raw = gr[h];
                float f_raw = gr[H   + h];
                float o_raw = gr[2*H + h];
                float g_raw = gr[3*H + h];

                float iv = 1.0f/(1.0f+expf(-i_raw));  /* sigmoid */
                float fv = 1.0f/(1.0f+expf(-f_raw));
                float ov = 1.0f/(1.0f+expf(-o_raw));
                float gv = tanhf(g_raw);

                l->i_gate[t]->data[b*H+h] = iv;
                l->f_gate[t]->data[b*H+h] = fv;
                l->o_gate[t]->data[b*H+h] = ov;
                l->g_gate[t]->data[b*H+h] = gv;

                /* c[t] = f*c[t-1] + i*g */
                float cv = fv * l->c_state[t]->data[b*H+h]
                         + iv * gv;
                l->c_state[t+1]->data[b*H+h] = cv;

                /* h[t] = o * tanh(c[t]) */
                float hv = ov * tanhf(cv);
                l->h_state[t+1]->data[b*H+h] = hv;

                /* write to output */
                output->data[b*(seq*H) + t*H + h] = hv;
            }
        }
    }

    tensor_free(gates_raw);
    tensor_free(xh_out);
    tensor_free(hh_out);
}

/* ── LSTM backward (BPTT) ────────────────────────────────────────── */

void lstm_backward(LSTMLayer *l, const Tensor *grad_out) {
    int batch = l->batch_cache;
    int seq   = l->seq_cache;
    int I     = l->input_size;
    int H     = l->hidden_size;
    int H4    = 4 * H;

    lstm_zero_grad(l);
    tensor_free(l->dX);
    int sh3[3] = {batch, seq, I};
    l->dX = tensor_zeros(sh3, 3);

    Tensor *dh_next = alloc2(batch, H);
    Tensor *dc_next = alloc2(batch, H);
    Tensor *d_gates = alloc2(batch, H4);
    Tensor *dx_t    = alloc2(batch, I);

    tensor_fill(dh_next, 0.0f);
    tensor_fill(dc_next, 0.0f);

    for (int t = seq-1; t >= 0; t--) {
        /* dh[t] = grad_out[t] + dh_next */
        for (int b = 0; b < batch; b++)
        for (int h = 0; h < H; h++) {
            float dh = grad_out->data[b*(seq*H)+t*H+h]
                     + dh_next->data[b*H+h];

            float ov = l->o_gate[t]->data[b*H+h];
            float cv = l->c_state[t+1]->data[b*H+h];
            float tc = tanhf(cv);
            float iv = l->i_gate[t]->data[b*H+h];
            float fv = l->f_gate[t]->data[b*H+h];
            float gv = l->g_gate[t]->data[b*H+h];

            /* dc[t] */
            float dc = dh * ov * (1.0f - tc*tc) + dc_next->data[b*H+h];

            /* gate gradients */
            float do_ = dh * tc;
            float di  = dc * gv;
            float df  = dc * l->c_state[t]->data[b*H+h];
            float dg  = dc * iv;

            /* dc[t-1] */
            dc_next->data[b*H+h] = dc * fv;

            /* raw gate gradients (through activation) */
            d_gates->data[b*H4+h]        = di * iv*(1.0f-iv); /* sigmoid */
            d_gates->data[b*H4+H+h]      = df * fv*(1.0f-fv);
            d_gates->data[b*H4+2*H+h]    = do_* ov*(1.0f-ov);
            d_gates->data[b*H4+3*H+h]    = dg * (1.0f-gv*gv); /* tanh  */
        }

        /* dW_ih += x[t]^T @ d_gates  → [H4, I] */
        for (int b = 0; b < batch; b++)
        for (int g = 0; g < H4; g++) {
            float dg = d_gates->data[b*H4+g];
            for (int i = 0; i < I; i++)
                l->dW_ih->data[g*I+i] += dg * l->x_cache[t]->data[b*I+i];
        }

        /* dW_hh += h[t]^T @ d_gates  → [H4, H] */
        for (int b = 0; b < batch; b++)
        for (int g = 0; g < H4; g++) {
            float dg = d_gates->data[b*H4+g];
            for (int h = 0; h < H; h++)
                l->dW_hh->data[g*H+h] += dg * l->h_state[t]->data[b*H+h];
        }

        /* db += sum over batch of d_gates */
        for (int b = 0; b < batch; b++)
        for (int g = 0; g < H4; g++)
            l->db->data[g] += d_gates->data[b*H4+g];

        /* dx[t] = d_gates @ W_ih   [batch,H4] @ [H4,I] → [batch,I] */
        tensor_fill(dx_t, 0.0f);
        for (int b = 0; b < batch; b++)
        for (int g = 0; g < H4; g++) {
            float dg = d_gates->data[b*H4+g];
            for (int i = 0; i < I; i++)
                dx_t->data[b*I+i] += dg * l->W_ih->data[g*I+i];
        }
        for (int b = 0; b < batch; b++)
        for (int i = 0; i < I; i++)
            l->dX->data[b*(seq*I)+t*I+i] = dx_t->data[b*I+i];

        /* dh_next = d_gates @ W_hh  [batch,H4] @ [H4,H] → [batch,H] */
        tensor_fill(dh_next, 0.0f);
        for (int b = 0; b < batch; b++)
        for (int g = 0; g < H4; g++) {
            float dg = d_gates->data[b*H4+g];
            for (int h = 0; h < H; h++)
                dh_next->data[b*H+h] += dg * l->W_hh->data[g*H+h];
        }
    }

    float scale = 1.0f / (float)(batch * seq);
    tensor_scale(l->dW_ih, scale, l->dW_ih);
    tensor_scale(l->dW_hh, scale, l->dW_hh);
    tensor_scale(l->db,    scale, l->db);

    tensor_free(dh_next); tensor_free(dc_next);
    tensor_free(d_gates); tensor_free(dx_t);
}

void lstm_zero_grad(LSTMLayer *l) {
    tensor_fill(l->dW_ih, 0.0f);
    tensor_fill(l->dW_hh, 0.0f);
    tensor_fill(l->db,    0.0f);
}

void lstm_update_sgd(LSTMLayer *l, float lr) {
    for (size_t i = 0; i < l->W_ih->size; i++)
        l->W_ih->data[i] -= lr * l->dW_ih->data[i];
    for (size_t i = 0; i < l->W_hh->size; i++)
        l->W_hh->data[i] -= lr * l->dW_hh->data[i];
    for (size_t i = 0; i < l->b->size; i++)
        l->b->data[i]    -= lr * l->db->data[i];
}

int lstm_param_count(const LSTMLayer *l) {
    return (int)(l->W_ih->size + l->W_hh->size + l->b->size);
}

/* ── gradient clipping (clip by global norm) ─────────────────────── */

static void clip_grads(Tensor **grads, int n, float max_norm) {
    float total = 0.0f;
    for (int i = 0; i < n; i++)
        for (size_t j = 0; j < grads[i]->size; j++)
            total += grads[i]->data[j] * grads[i]->data[j];
    float norm = sqrtf(total);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (int i = 0; i < n; i++)
            for (size_t j = 0; j < grads[i]->size; j++)
                grads[i]->data[j] *= scale;
    }
}

void lstm_clip_gradients(LSTMLayer *l, float max_norm) {
    Tensor *gs[3] = {l->dW_ih, l->dW_hh, l->db};
    clip_grads(gs, 3, max_norm);
}

void rnn_clip_gradients(RNNLayer *l, float max_norm) {
    Tensor *gs[3] = {l->dW_xh, l->dW_hh, l->db_h};
    clip_grads(gs, 3, max_norm);
}
