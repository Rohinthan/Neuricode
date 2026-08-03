#ifndef NEURALC_MODEL_H
#define NEURALC_MODEL_H

/*
 * model.h — reusable inference-time Model abstraction for NeuralC.
 *
 * This is the seam between "a loaded Network" and "a program that
 * wants to generate tokens": nothing in this header knows about argv,
 * stdout, or CLI concerns. A Model owns every buffer it needs
 * (recurrent state, one-hot scratch, dense output) so callers never
 * allocate anything on the hot path.
 *
 * Supported architecture (validated at load time, see model.c):
 *   [0] LAYER_RNN or LAYER_LSTM  — recurrent encoder
 *   [1] LAYER_DENSE (ACT_SOFTMAX) — output projection to vocabulary
 *
 * Thread-safety: a single Model* is NOT safe to use from more than one
 * thread concurrently (it carries mutable recurrent state that
 * model_step advances in place). Separate Model* instances (e.g. two
 * model_load() calls on the same file) are fully independent and may
 * be driven from different threads simultaneously.
 */

#include <stddef.h>
#include "sampler.h"

typedef struct Model Model;

/* ── lifecycle ──────────────────────────────────────────────────── */

/*
 * model_load: opens and validates a model file (nn_save_model()
 * format) at `path`, then allocates every inference-time scratch
 * buffer once, up front.
 *
 * On failure, returns NULL and — if err/err_len are non-NULL/non-zero
 * — writes a NUL-terminated human-readable message into `err`
 * (truncated to fit if necessary). Passing err=NULL/err_len=0 is
 * valid if the caller doesn't want a message.
 *
 * Failure modes: file not found/unreadable, corrupt/truncated model
 * file, or an architecture this runtime doesn't support (anything
 * other than exactly [RNN|LSTM] -> Dense(softmax), or a dimension
 * mismatch between the two layers).
 */
Model *model_load(const char *path, char *err, size_t err_len);

/* Frees every buffer owned by `m`, including the underlying Network.
 * Safe to call with m == NULL (no-op). */
void model_free(Model *m);

/* Resets recurrent hidden/cell state to zero-init. Call this before
 * starting a new, unrelated sequence on an already-loaded Model —
 * model_load() itself starts a Model in the reset state, so this is
 * only needed for reuse across multiple independent generations. */
void model_reset_state(Model *m);

/* ── inference ──────────────────────────────────────────────────── */

/*
 * model_step: feeds one token through the recurrent encoder + dense
 * head, advancing internal state in place, and writes the resulting
 * probability distribution (length model_vocab_size(m)) into
 * out_probs (caller-owned buffer).
 *
 * token_id must be in [0, model_vocab_size(m)). No allocation happens
 * inside this call. Returns 0 on success, -1 on an out-of-range
 * token_id.
 */
int model_step(Model *m, int token_id, float *out_probs);

/*
 * model_generate: warms up on `prompt` (prompt_len raw bytes, each
 * must be < model_vocab_size(m)) with no output, then samples up to
 * max_tokens tokens using `sampler`, invoking cb(token, user_data) for
 * each sampled token as soon as it's produced (streaming — no
 * internal buffering of generated output).
 *
 * If eos_token >= 0, generation stops early (before max_tokens) the
 * first time that token is sampled; cb() is still invoked for it.
 * Pass eos_token = -1 to disable early stopping.
 *
 * Does NOT reset state first — call model_reset_state() beforehand if
 * this should start a fresh sequence rather than continue from
 * wherever the Model currently is.
 *
 * Returns 0 on success, -1 if the prompt contains a byte outside the
 * model's vocabulary.
 */
int model_generate(Model *m, const unsigned char *prompt, size_t prompt_len,
                    int max_tokens, int eos_token, const Sampler *sampler,
                    void (*cb)(int token, void *user_data), void *user_data);

/* ── introspection ──────────────────────────────────────────────── */

int model_vocab_size(const Model *m);
int model_hidden_size(const Model *m);

#endif /* NEURALC_MODEL_H */
