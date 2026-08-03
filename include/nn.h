#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "layer.h"
#include "dropout.h"
#include "batchnorm.h"
#include "rnn.h"

#define NN_MAX_LAYERS 32

/* ══════════════════════════════════════════════════════════════════
 * INTERFACE NOTES — how nn.c talks to each concrete layer module.
 * ══════════════════════════════════════════════════════════════════
 * The five modules are NOT uniform (unlike a from-scratch design
 * would be) — each has its own shape convention and its own way of
 * returning "gradient w.r.t. my input". nn.c's switch-case dispatch
 * bridges these differences case by case rather than assuming one
 * shared contract:
 *
 *   DenseLayer   — 2D [batch, in_features] -> [batch, out_features].
 *                  backward(l, grad_out) populates l->dX internally;
 *                  no training/eval distinction.
 *   DropoutLayer — any shape, output shape == input shape exactly.
 *                  No in/out feature count is stored on the struct at
 *                  all (it's shape-agnostic). Mode is a persistent
 *                  flag flipped via dropout_train()/dropout_eval(),
 *                  not a forward() parameter. backward() takes a
 *                  caller-allocated grad_in (no internal dX).
 *   BatchNorm    — 2D [batch, num_features], output shape == input
 *                  shape. Mode via batchnorm_train()/batchnorm_eval().
 *                  backward() also takes a caller-allocated grad_in.
 *   RNNLayer     — 3D [batch, seq_len, input_size] ->
 *                  [batch, seq_len, hidden_size]. forward() takes an
 *                  explicit initial hidden state (NULL = zeros).
 *                  backward(l, grad_out) populates l->dX internally.
 *   LSTMLayer    — same 3D shape convention as RNNLayer, forward()
 *                  takes explicit initial hidden *and* cell state
 *                  (NULL = zeros for either/both). backward(l,
 *                  grad_out) populates l->dX internally.
 *
 * Because RNN/LSTM forward() accepts external initial state but
 * nn_forward() below has nowhere to persist a hidden/cell state
 * across separate nn_forward() calls, every nn_forward() call feeds
 * RNN/LSTM layers a fresh zero initial state (h_init = c_init = NULL).
 * That means a Network is a *stateless-per-call* sequence processor:
 * fine for "encode this whole sequence, get its outputs" style use,
 * but NOT a mechanism for streaming state across many short calls. If
 * you need persistent state across calls, drive that RNNLayer/
 * LSTMLayer directly instead of through Network.
 *
 * Because Dropout/BatchNorm need a caller-allocated grad_in buffer
 * (they have no dX field of their own), Network caches one scratch
 * buffer per such layer in grad_cache[] — sized to match that layer's
 * own (shape-preserving) output buffer, and reused across steps the
 * same way outputs[] is.
 * ════════════════════════════════════════════════════════════════ */

/* ── layer kind tag ────────────────────────────────────────────────
 * One entry per concrete layer module linked into libneuralc. Stored
 * as an int32 in saved model files, so treat existing values as a
 * stable ABI once shipped — append new kinds at the end, never
 * renumber or remove one, or old .bin files stop loading correctly. */
typedef enum {
    LAYER_DENSE     = 0,
    LAYER_DROPOUT   = 1,
    LAYER_BATCHNORM = 2,
    LAYER_RNN       = 3,
    LAYER_LSTM      = 4
} LayerType;

/* ── generic layer container ──────────────────────────────────────
 * `layer_ptr` is opaque outside of the type-dispatch switch blocks in
 * nn.c: it must always be cast back to the concrete struct indicated
 * by `type` before use — DenseLayer*, DropoutLayer*, BatchNorm*,
 * RNNLayer*, or LSTMLayer*. Never dereference it directly. */
typedef struct {
    LayerType type;
    void     *layer_ptr;
} NetworkLayer;

/* ── loss functions ─────────────────────────────────────────────── */
typedef enum {
    LOSS_MSE             = 0,   /* mean squared error          */
    LOSS_BINARY_CROSS    = 1,   /* binary cross-entropy        */
    LOSS_CROSS_ENTROPY   = 2    /* softmax + cross-entropy     */
} LossType;

/* ── network ────────────────────────────────────────────────────── */
typedef struct {
    NetworkLayer layers[NN_MAX_LAYERS];
    int          num_layers;

    /* Cached per-layer forward output (2D for Dense/Dropout/
     * BatchNorm, 3D [batch,seq_len,hidden] for RNN/LSTM). */
    Tensor      *outputs[NN_MAX_LAYERS];

    /* Cached per-layer grad_in scratch buffer, used only for
     * LAYER_DROPOUT / LAYER_BATCHNORM during nn_backward() (Dense/
     * RNN/LSTM already own their gradient in l->dX). NULL/unused for
     * every other layer type. */
    Tensor      *grad_cache[NN_MAX_LAYERS];
} Network;

/* ── lifecycle ──────────────────────────────────────────────────── */
Network *nn_create(void);

/* Generic append — stores the (type, pointer) pair directly. Prefer
 * the nn_add_<kind> convenience wrappers below at call sites; they're
 * thin, self-documenting aliases over this one. */
void nn_add_layer(Network *net, LayerType type, void *layer_ptr);

void nn_add_dense    (Network *net, DenseLayer   *layer);
void nn_add_dropout  (Network *net, DropoutLayer *layer);
void nn_add_batchnorm(Network *net, BatchNorm    *layer);
void nn_add_rnn      (Network *net, RNNLayer     *layer);
void nn_add_lstm     (Network *net, LSTMLayer    *layer);

void nn_free(Network *net);

/* ── forward / backward / loss ──────────────────────────────────── */
/*
 * nn_forward:
 *   input     shape depends on the first layer (2D for Dense/Dropout/
 *             BatchNorm-first networks, 3D [batch,seq,feat] if the
 *             first layer is RNN/LSTM).
 *   output    caller-allocated, must match the last layer's output
 *             size exactly.
 *   training  1 during training, 0 at inference. Threaded into
 *             Dropout/BatchNorm via dropout_train()/dropout_eval()
 *             and batchnorm_train()/batchnorm_eval() before their
 *             forward() runs (those two have no forward()-level
 *             training flag of their own — it's persistent state on
 *             the layer). Has no effect on Dense/RNN/LSTM.
 * Dispatches to each layer's *_forward() through a switch on ->type.
 */
void nn_forward(Network *net, const Tensor *input, Tensor *output, int training);

/*
 * nn_loss:  compute scalar loss + initial gradient
 *   pred    [batch, out] — network output
 *   target  [batch, out] — ground truth
 *   grad    [batch, out] — dL/d(pred) written here (caller allocates)
 *   returns scalar loss value
 */
float nn_loss(LossType type, const Tensor *pred, const Tensor *target,
              Tensor *grad);

/*
 * nn_backward:
 *   grad_out  — gradient of loss w.r.t. last layer's output, matching
 *               that layer's output shape (2D or 3D, see nn_forward).
 *   Runs backprop through all layers (switch-dispatched per ->type).
 *   Dense/RNN/LSTM: calls layer_backward(l, g), then continues with
 *     g = l->dX.
 *   Dropout/BatchNorm: calls layer_backward(l, g, grad_cache[i])
 *     (they need a caller-supplied destination), then continues with
 *     g = grad_cache[i].
 */
void nn_backward(Network *net, const Tensor *grad_out);

/* ── convenience: single training step ─────────────────────────── */
/*
 * Returns the scalar loss value.
 * Does NOT update weights (call optimizer after this).
 */
float nn_train_step(Network *net, const Tensor *input, const Tensor *target,
                    LossType loss_type, Tensor *pred_buf, Tensor *grad_buf);

/* ── utilities ──────────────────────────────────────────────────── */
void nn_print_summary(const Network *net);
int  nn_total_params(const Network *net);

/* ── metrics ────────────────────────────────────────────────────── */
/*
 * nn_accuracy_binary:
 *   pred   [batch, 1]  sigmoid output (threshold 0.5)
 *   target [batch, 1]  ground truth 0 or 1
 *   Returns fraction correct in [0.0, 1.0]
 */
float nn_accuracy_binary(const Tensor *pred, const Tensor *target);

/*
 * nn_accuracy_multiclass:
 *   pred   [batch, classes]  softmax output
 *   target [batch, classes]  one-hot ground truth
 *   Returns fraction correct in [0.0, 1.0]
 */
float nn_accuracy_multiclass(const Tensor *pred, const Tensor *target);

/* ── unified model serialization ──────────────────────────────────
 * Binary layout (all integers host-endian int32, all floats
 * host-endian float32 — not cross-endian-portable, matching the rest
 * of libneuralc's raw fwrite/fread usage):
 *
 *   [ uint32   magic       ]  = NN_MODEL_MAGIC, validated on load
 *   [ int32    version     ]  = NN_MODEL_VERSION
 *   [ int32    num_layers  ]
 *   for each layer, in order:
 *     [ int32  LayerType                 ]
 *     [ type-specific config ints/floats ]  (exactly what that type's
 *                                             *_create() factory needs)
 *     [ type-specific parameter blocks   ]  (raw fwrite of every
 *                                             learnable tensor's
 *                                             ->data, fixed order)
 *
 * Per-type config + params actually written (see nn.c for the exact
 * per-field order):
 *   LAYER_DENSE:     [in_features, out_features, activation]   ->  W, b
 *   LAYER_DROPOUT:   [drop_prob]                                -> (none)
 *   LAYER_BATCHNORM: [num_features, eps, momentum]             ->  gamma, beta, running_mean, running_var
 *   LAYER_RNN:       [input_size, hidden_size]                 ->  W_xh, W_hh, b_h
 *   LAYER_LSTM:      [input_size, hidden_size]                 ->  W_ih, W_hh, b
 *
 * nn_save_model: walks `net`'s existing layers and writes the stream
 *   above. Does not modify `net`.
 * nn_load_model: reads the stream and rebuilds `net` from scratch via
 *   an abstract-factory switch (any existing layers already in `net`
 *   are freed first, so this also works as "replace current model").
 *   Every fread is count-checked and every decoded dimension is
 *   range-validated before it's used to size any allocation, so a
 *   truncated or corrupted file fails cleanly with a returned error
 *   code instead of a segfault. On any failure, whatever layers had
 *   already been constructed during that same load attempt are freed
 *   and `net` is left in a valid, empty state.
 * Both return 0 on success, -1 on failure. */
#define NN_MODEL_MAGIC   0x43464E4Eu  /* ASCII "CFNN" */
#define NN_MODEL_VERSION 1

int nn_save_model(Network *net, const char *path);
int nn_load_model(Network *net, const char *path);

/* ── legacy weight I/O ─────────────────────────────────────────────
 * Thin backward-compatible aliases kept for existing call sites; both
 * now forward to the unified model (de)serializer above, which works
 * for heterogeneous networks too — prefer the *_model() names in new
 * code. */
int  nn_save(const Network *net, const char *path);
int  nn_load(Network *net, const char *path);

#endif /* NN_H */
