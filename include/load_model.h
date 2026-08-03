#ifndef LOAD_MODEL_H
#define LOAD_MODEL_H

#include <stdint.h>

/*
 * load_model.h — runtime Model struct and loader API for the custom
 * NCRN binary format (see load_model.c for the exact on-disk layout
 * and validation rules).
 *
 * Only one of the two weight sets is ever populated, selected by
 * model_type:
 *   model_type == 0 (RNN)  -> Wx, Wh, b are non-NULL;
 *                              Wx_lstm/Wh_lstm/b_lstm are NULL.
 *   model_type == 1 (LSTM) -> Wx_lstm, Wh_lstm, b_lstm are non-NULL;
 *                              Wx/Wh/b are NULL.
 * W_dense/b_dense are always populated for both types.
 *
 * All weight arrays are flat, row-major float32 buffers matching the
 * shapes documented in load_model.c:
 *   RNN:   Wx [input_size * hidden_size]
 *          Wh [hidden_size * hidden_size]
 *          b  [hidden_size]
 *   LSTM:  Wx_lstm [4 * input_size  * hidden_size]   gate order [i,f,g,o]
 *          Wh_lstm [4 * hidden_size * hidden_size]
 *          b_lstm  [4 * hidden_size]
 *   Dense: W_dense [hidden_size * vocab_size]   (in_features * out_features)
 *          b_dense [vocab_size]
 */

typedef struct {
    uint32_t model_type;    /* 0 = RNN, 1 = LSTM   */
    uint32_t vocab_size;    /* always 256          */
    uint32_t hidden_size;

    /* RNN weights — NULL unless model_type == 0 */
    float *Wx, *Wh, *b;

    /* LSTM weights — NULL unless model_type == 1 */
    float *Wx_lstm, *Wh_lstm, *b_lstm;

    /* Dense output layer — always populated */
    float *W_dense, *b_dense;
} Model;

/*
 * load_model: opens `path`, reads and validates the ModelHeader, then
 * reads the recurrent layer (RNN or LSTM per model_type) and the
 * Dense output layer, in that fixed sequential order (no seeking).
 *
 * Returns a fully populated Model* on success.
 * Returns NULL on any failure (bad magic/version/vocab_size/
 * model_type, dimension mismatch, short read, or allocation failure)
 * — a clear message is printed to stderr and everything allocated so
 * far is freed before returning, so there is nothing for the caller
 * to clean up on failure.
 */
Model *load_model(const char *path);

/*
 * free_model: releases every buffer owned by `m`, then `m` itself.
 * Safe to call with m == NULL (no-op). Safe to call on a partially
 * populated Model (as load_model() itself does internally on error
 * paths) since unused pointer fields are always NULL and free(NULL)
 * is a no-op.
 */
void free_model(Model *m);

#endif /* LOAD_MODEL_H */
