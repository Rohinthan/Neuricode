#ifndef NEURICODE_TRANSFORMER_H
#define NEURICODE_TRANSFORMER_H

#include <stddef.h>
#include <stdbool.h>

#include <stdint.h>

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

#ifndef MODEL_TYPE_RNN
#define MODEL_TYPE_RNN          0
#endif
#ifndef MODEL_TYPE_TRANSFORMER
#define MODEL_TYPE_TRANSFORMER   1
#endif
#ifndef ACTIVE_MODEL_TYPE
#define ACTIVE_MODEL_TYPE       MODEL_TYPE_TRANSFORMER
#endif

#ifndef TRANS_VOCAB_SIZE
#define TRANS_VOCAB_SIZE 32000
#endif
#ifndef TRANS_DIM
#define TRANS_DIM 768
#endif
#ifndef TRANS_HIDDEN_DIM
#define TRANS_HIDDEN_DIM 2048
#endif
#ifndef TRANS_N_LAYERS
#define TRANS_N_LAYERS 12
#endif
#ifndef TRANS_N_HEADS
#define TRANS_N_HEADS 12
#endif
#ifndef TRANS_MAX_SEQ_LEN
#define TRANS_MAX_SEQ_LEN 2048
#endif

typedef struct {
    int vocab_size;   // Default: 32000
    int dim;          // Default: 768
    int hidden_dim;   // Default: 2048
    int n_layers;     // Default: 12
    int n_heads;      // Default: 12
    int seq_len;      // Default: 2048
} TransformerConfig;

typedef struct {
    float *token_embedding_table; // [vocab_size, dim]
    float *rms_att_weight;        // [n_layers, dim]
    float *wq;                     // [n_layers, dim, dim]
    float *wk;                     // [n_layers, dim, dim]
    float *wv;                     // [n_layers, dim, dim]
    float *wo;                     // [n_layers, dim, dim]
    float *rms_ffn_weight;        // [n_layers, dim]
    float *w1;                     // [n_layers, hidden_dim, dim] (SwiGLU gate)
    float *w2;                     // [n_layers, dim, hidden_dim] (SwiGLU down)
    float *w3;                     // [n_layers, hidden_dim, dim] (SwiGLU up)
    float *rms_final_weight;      // [dim]
    float *wcls;                   // [vocab_size, dim]
} TransformerWeights;

typedef struct {
    // KV-Cache optimized for sequential cache locality: [n_layers, n_heads, seq_len, head_size]
    float *key_cache;   // Pre-allocated contiguous KV cache
    float *value_cache; // Pre-allocated contiguous KV cache
    float *x;           // State vector [dim]
    float *xb;          // Work buffer [dim]
    float *xb2;         // Work buffer [dim]
    float *hb;          // FFN hidden buffer [hidden_dim]
    float *hb2;         // FFN hidden buffer [hidden_dim]
    float *q;           // Query buffer [dim]
    float *k;           // Key buffer [dim]
    float *v;           // Value buffer [dim]
    float *att;         // Attention scores buffer [n_heads, seq_len]
    float *logits;      // Output logits [vocab_size]
} TransformerState;

// Precomputed RoPE Frequency Lookup Table
typedef struct {
    float *cos;         // Precomputed cos: [seq_len, head_size / 2]
    float *sin;         // Precomputed sin: [seq_len, head_size / 2]
} RoPETable;

// Quantized INT8 Weight Representation (Q8_0)
typedef struct {
    int8_t *qweights;   // Quantized signed int8 weights
    float  *scales;     // Per-row scaling factors
    size_t rows;
    size_t cols;
} QuantizedWeightQ8;

typedef struct {
    TransformerConfig config;
    TransformerWeights weights;
    TransformerState state;
    RoPETable rope;
    bool debug_mode;
} TransformerModel;

// Core Lifecycle API
TransformerModel *transformer_init(TransformerConfig config);
TransformerModel *transformer_init_from_config(void);
void transformer_free(TransformerModel *model);

// Inference API
float *transformer_forward(TransformerModel *model, int token, int pos);
float *transformer_forward_batch(TransformerModel *model, const int *tokens, int n_tokens, int start_pos);

// Binary Model I/O
int transformer_load_weights(TransformerModel *model, const char *filepath);
int transformer_save_weights(const TransformerModel *model, const char *filepath);

// Debugging & Validation API
void transformer_set_debug(TransformerModel *model, bool enable);
bool validate_tensor_buffer(const float *data, size_t size, const char *name);

// INT8 Quantization Utilities
QuantizedWeightQ8 *quantize_weight_q8(const float *src, size_t rows, size_t cols);
void free_quantized_weight_q8(QuantizedWeightQ8 *qw);
void matmul_q8(float *xout, const float *x, const QuantizedWeightQ8 *qw, int n, int d);

#endif // NEURICODE_TRANSFORMER_H
