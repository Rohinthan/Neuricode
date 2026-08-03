/*
 * demo_char_rnn.c — Character-level text generation with a vanilla RNN
 *
 * Architecture:  one-hot(vocab) -> RNN(hidden=128) -> Dense(vocab, softmax)
 * Loss:          Cross-entropy (per character, per timestep)
 * Optimizer:     Plain SGD with gradient clipping
 *
 * ── Why this drives RNNLayer/DenseLayer directly instead of Network ──
 * nn_forward()'s LAYER_DENSE case hard-requires 2D [batch, features]
 * input (CF_CHECK, hard exit on failure), but RNN/LSTM produce 3D
 * [batch, seq_len, hidden]. There's no bridging case for RNN -> Dense
 * in nn.c's switch dispatch, so chaining them through Network would
 * crash the instant nn_forward reached the Dense layer. nn.h's own
 * comments say as much: "If you need persistent state across calls,
 * drive that RNNLayer/LSTMLayer directly instead of through Network."
 * That's exactly what this file does — the 3D<->2D bridge is a single
 * tensor_reshape() each way (a view, not a copy — collapsing
 * [batch,seq,hidden] to [batch*seq,hidden] doesn't reorder anything
 * since both are row-major contiguous).
 *
 * ── Training simplification, stated plainly ──
 * Each training step samples a batch of independent, fixed-length
 * windows from the corpus and runs them with a zero initial hidden
 * state (matching nn_forward's own RNN convention: "every call feeds
 * RNN/LSTM layers a fresh zero initial state"). This is NOT full-
 * corpus BPTT with hidden state carried across steps — it's simpler,
 * on purpose, for a self-contained example. Text GENERATION below
 * does carry hidden state character-by-character within one sample,
 * since that's what makes generated text coherent.
 *
 * ── Known library bug, worked around locally in this file ──
 * nn_loss()'s LOSS_CROSS_ENTROPY case (nn.c) and dense_backward()'s
 * ACT_SOFTMAX case (layer.c) both divide by batch, so Dense's dW/db
 * end up divided by batch TWICE (1/batch^2 instead of 1/batch) —
 * gradients ~batch times too small, training effectively stalled.
 * Corrected locally, right after the dense_backward() call below,
 * rather than touched in the library itself (deliberate choice — see
 * conversation this file came from). dX is unaffected by the bug and
 * is left alone; only dW/db are corrected.
 *
 * Build:  make demo_char_rnn
 * Run:    ./demo_char_rnn
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

#include "tensor.h"
#include "layer.h"
#include "nn.h"
#include "rnn.h"

/* ═══════════════════════════════════════════════════════════════════
 *  Hyperparameters
 * ═══════════════════════════════════════════════════════════════════ */
#define HIDDEN_SIZE   128
#define SEQ_LEN       25      /* well under RNN_MAX_SEQ (512)          */
#define BATCH_SIZE    16
#define TRAIN_STEPS   3000
#define LEARNING_RATE 0.05f
#define GRAD_CLIP_NORM 5.0f
#define PRINT_EVERY   250
#define SAMPLE_LEN    200

/* Embedded training text — keeps this demo runnable with zero setup,
 * no dataset download required (unlike demo_mnist.c's MNIST files).
 * Swap in a real corpus via argv[1] (plain text file) for anything
 * beyond a sanity check — this snippet is far too small to produce
 * genuinely fluent output, only to prove the pipeline works. */
static const char *DEFAULT_TEXT =
    "the quick brown fox jumps over the lazy dog. "
    "a journey of a thousand miles begins with a single step. "
    "all that glitters is not gold. "
    "actions speak louder than words. "
    "the pen is mightier than the sword. "
    "practice makes perfect. "
    "where there is a will there is a way. "
    "knowledge is power. "
    "time and tide wait for no man. "
    "the early bird catches the worm. ";

/* ═══════════════════════════════════════════════════════════════════
 *  Vocabulary: a simple 256-entry lookup, since char is one byte
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    int  char_to_idx[256];   /* -1 if character never appears in corpus */
    char idx_to_char[256];
    int  vocab_size;
} Vocab;

static void vocab_build(Vocab *v, const char *text) {
    for (int i = 0; i < 256; i++) v->char_to_idx[i] = -1;
    v->vocab_size = 0;

    for (const char *p = text; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (v->char_to_idx[c] == -1) {
            v->char_to_idx[c] = v->vocab_size;
            v->idx_to_char[v->vocab_size] = (char)c;
            v->vocab_size++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  Manual SGD update for a Dense layer's parameters
 * ═══════════════════════════════════════════════════════════════════
 * Not going through Network/SGD here since this file deliberately
 * bypasses Network entirely (see file header) — a plain elementwise
 * update is all Dense needs after dense_backward() has populated
 * l->dW/l->db, and keeps the RNN and Dense update code symmetric
 * (RNN already ships its own rnn_update_sgd() in rnn.h).
 */
static void dense_sgd_update(DenseLayer *l, float lr) {
    for (size_t i = 0; i < l->W->size; i++)
        l->W->data[i] -= lr * l->dW->data[i];
    for (size_t i = 0; i < l->b->size; i++)
        l->b->data[i] -= lr * l->db->data[i];
}

/* ═══════════════════════════════════════════════════════════════════
 *  Build one training batch: BATCH_SIZE random windows of SEQ_LEN
 *  characters, one-hot encoded.
 *    X [BATCH_SIZE, SEQ_LEN, vocab]        — input characters
 *    Y [BATCH_SIZE*SEQ_LEN, vocab]         — next-character targets,
 *                                             pre-flattened to match
 *                                             the Dense layer's 2D
 *                                             output shape directly
 * ═══════════════════════════════════════════════════════════════════ */
static void make_batch(const char *text, int text_len, const Vocab *v,
                       Tensor *X, Tensor *Y) {
    tensor_fill(X, 0.0f);
    tensor_fill(Y, 0.0f);
    int vocab = v->vocab_size;

    for (int b = 0; b < BATCH_SIZE; b++) {
        int start = rand() % (text_len - SEQ_LEN - 1);
        for (int t = 0; t < SEQ_LEN; t++) {
            int in_idx  = v->char_to_idx[(unsigned char)text[start + t]];
            int out_idx = v->char_to_idx[(unsigned char)text[start + t + 1]];

            /* X[b, t, in_idx] = 1.0 */
            X->data[(size_t)b*SEQ_LEN*vocab + (size_t)t*vocab + in_idx] = 1.0f;

            /* Y is pre-flattened to [BATCH_SIZE*SEQ_LEN, vocab] so it
             * matches the Dense layer's output shape directly — row
             * (b*SEQ_LEN + t) corresponds to X's [b, t, :]. */
            size_t row = (size_t)b*SEQ_LEN + t;
            Y->data[row*vocab + out_idx] = 1.0f;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  Sample generated text from the current model, carrying hidden
 *  state character-by-character (unlike training's zero-h_init
 *  batches — coherent generation needs the real running state).
 * ═══════════════════════════════════════════════════════════════════ */
static void generate_sample(RNNLayer *rnn, DenseLayer *out_layer,
                            const Vocab *v, char seed, int length) {
    int vocab = v->vocab_size;
    int sh_in[3]  = {1, 1, vocab};   /* batch=1, seq=1 */
    int sh_out[3] = {1, 1, HIDDEN_SIZE};
    int sh_dense_in[2]  = {1, HIDDEN_SIZE};
    int sh_dense_out[2] = {1, vocab};

    Tensor *x         = tensor_zeros(sh_in, 3);
    Tensor *rnn_out   = tensor_zeros(sh_out, 3);
    Tensor *dense_out = tensor_zeros(sh_dense_out, 2);

    Tensor *h_prev = NULL;   /* NULL = zero initial hidden state */
    char c = seed;
    printf("\"");
    putchar(c);

    for (int step = 0; step < length; step++) {
        tensor_fill(x, 0.0f);
        int idx = v->char_to_idx[(unsigned char)c];
        if (idx < 0) idx = 0;   /* unseen char fallback */
        x->data[idx] = 1.0f;

        rnn_forward(rnn, x, h_prev, rnn_out);
        /* rnn->h_states[1] is this call's resulting hidden state
         * (h_states[0] is whatever h_init was, h_states[1..T] are the
         * per-timestep outputs — see rnn.h) — reuse it as next h_init
         * to carry state across generated characters. */
        h_prev = rnn->h_states[1];

        /* reshape [1,1,hidden] view -> [1,hidden] for Dense (view,
         * must be freed separately from the underlying buffer) */
        Tensor *rnn_out_2d = tensor_reshape(rnn_out, sh_dense_in, 2);
        dense_forward(out_layer, rnn_out_2d, dense_out);
        tensor_free(rnn_out_2d);

        /* greedy sampling (argmax) — simple and deterministic; for
         * more varied output, sample from dense_out's probabilities
         * instead (dense_out is already a softmax distribution) */
        int next_idx = tensor_argmax(dense_out);
        c = v->idx_to_char[next_idx];
        putchar(c);
    }
    printf("\"\n");

    tensor_free(x);
    tensor_free(rnn_out);
    tensor_free(dense_out);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Demo-local OpenMP thread setup
 * ═══════════════════════════════════════════════════════════════════
 * neuralc_init.c's __attribute__((constructor)) already runs before
 * main() and sets the thread count from neuralc_config.h. This
 * function re-applies the SAME config values explicitly, from inside
 * this demo's main() — so demo_char_rnn's thread count is guaranteed
 * to track NEURALC_USE_OMP / NEURALC_OMP_AUTO / NEURALC_OMP_THREADS
 * even if the constructor path is ever skipped, reordered, or
 * shadowed by another translation unit's constructor. Only touches
 * this file — neuralc_init.c and the Makefile are untouched. */
static void demo_configure_threads(void) {
#ifdef USE_OMP
    long hw_cores  = sysconf(_SC_NPROCESSORS_ONLN);
    int  max_cores = (hw_cores > 0) ? (int)hw_cores : 1;

    int target_threads = max_cores;   /* default: AUTO = all cores */
    const char *mode = "AUTO";

#ifdef NEURALC_HAS_CONFIG
    if (!NEURALC_USE_OMP) {
        target_threads = 1;
        mode = "DISABLED (config)";
    } else if (!NEURALC_OMP_AUTO) {
        target_threads = NEURALC_OMP_THREADS;
        mode = "MANUAL (config)";
    }
    /* else: config says AUTO — target_threads stays max_cores */
#else
    /* No neuralc_config.h at build time — fall back to env var,
     * same as neuralc_init.c does in this case. */
    const char *env = getenv("OMP_NUM_THREADS");
    if (env) {
        int parsed = atoi(env);
        if (parsed > 0) {
            target_threads = parsed;
            mode = "MANUAL (env)";
        }
    }
#endif

    /* core oversubscription guard — never spawn more threads than
     * physical cores available */
    if (target_threads > max_cores) target_threads = max_cores;
    if (target_threads < 1) target_threads = 1;

    omp_set_num_threads(target_threads);
    printf("[demo_char_rnn] OpenMP threads: %d (%s, %d cores available)\n\n",
           target_threads, mode, max_cores);
#else
    printf("[demo_char_rnn] Built without OpenMP (-DUSE_OMP not set) — "
           "running single-threaded.\n\n");
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 *  Model checkpoint I/O
 * ═══════════════════════════════════════════════════════════════════
 * Writes the trained parameters to a flat binary file so a separate
 * process (pipeline.c) can load them for inference without repeating
 * training. This is a simple same-machine checkpoint format — raw
 * native-endian float dumps, not a portable/cross-platform format.
 *
 * File layout:
 *   int32_t  vocab_size
 *   int32_t  hidden_size
 *   float[]  rnn->W_xh->data       (hidden_size * vocab_size)   [hidden, vocab]
 *   float[]  rnn->W_hh->data       (hidden_size * hidden_size)  [hidden, hidden]
 *   float[]  rnn->b_h->data        (hidden_size)
 *   float[]  out_layer->W->data   (hidden_size * vocab_size)
 *   float[]  out_layer->b->data   (vocab_size)
 *
 * vocab_size/hidden_size are written purely so a loader can check its
 * own tokenizer vocab and HIDDEN_SIZE against what this file was
 * actually trained with, instead of silently reading floats into
 * mismatched buffers (wrong sizes, or worse, no crash but garbage
 * output).
 */
static int write_tensor(FILE *f, const Tensor *t) {
    size_t n = fwrite(t->data, sizeof(float), t->size, f);
    return (n == t->size) ? 0 : -1;
}

static int save_model(const char *path, const RNNLayer *rnn,
                       const DenseLayer *out_layer,
                       int vocab_size, int hidden_size) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int32_t vs = (int32_t)vocab_size;
    int32_t hs = (int32_t)hidden_size;
    int ok = 1;
    ok = ok && (fwrite(&vs, sizeof vs, 1, f) == 1);
    ok = ok && (fwrite(&hs, sizeof hs, 1, f) == 1);
    ok = ok && (write_tensor(f, rnn->W_xh) == 0);
    ok = ok && (write_tensor(f, rnn->W_hh) == 0);
    ok = ok && (write_tensor(f, rnn->b_h)  == 0);
    ok = ok && (write_tensor(f, out_layer->W) == 0);
    ok = ok && (write_tensor(f, out_layer->b) == 0);

    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    demo_configure_threads();

    printf("=== neuralc Character-Level RNN Text Generation ===\n\n");

    /* ── load corpus: file path via argv[1], else embedded sample ── */
    char *text = NULL;
    int   text_len = 0;

    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) {
            fprintf(stderr, "Could not open %s, using embedded sample text.\n", argv[1]);
        } else {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            text = (char *)malloc((size_t)sz + 1);
            size_t n = fread(text, 1, (size_t)sz, f);
            text[n] = '\0';
            text_len = (int)n;
            fclose(f);
            printf("Loaded corpus: %s (%d chars)\n", argv[1], text_len);
        }
    }
    if (!text) {
        text_len = (int)strlen(DEFAULT_TEXT);
        text = (char *)malloc((size_t)text_len + 1);
        memcpy(text, DEFAULT_TEXT, (size_t)text_len + 1);
        printf("Using embedded sample text (%d chars). Pass a text file "
               "path as argv[1] for a real corpus.\n", text_len);
    }
    if (text_len <= SEQ_LEN + 1) {
        fprintf(stderr, "Corpus too short for SEQ_LEN=%d\n", SEQ_LEN);
        return 1;
    }

    /* ── vocabulary ── */
    Vocab vocab;
    vocab_build(&vocab, text);
    printf("Vocabulary size: %d unique characters\n\n", vocab.vocab_size);

    /* ── model ── */
    RNNLayer   *rnn = rnn_create(vocab.vocab_size, HIDDEN_SIZE);
    DenseLayer *out_layer = dense_create(HIDDEN_SIZE, vocab.vocab_size, ACT_SOFTMAX);
    printf("Model: RNN(%d -> %d) -> Dense(%d -> %d, softmax)\n",
           vocab.vocab_size, HIDDEN_SIZE, HIDDEN_SIZE, vocab.vocab_size);
    printf("Params: RNN=%d  Dense=%d  Total=%d\n\n",
           rnn_param_count(rnn), dense_param_count(out_layer),
           rnn_param_count(rnn) + dense_param_count(out_layer));

    /* ── batch buffers, reused every step ── */
    int shX[3] = {BATCH_SIZE, SEQ_LEN, vocab.vocab_size};
    int shY[2] = {BATCH_SIZE * SEQ_LEN, vocab.vocab_size};
    int shRnnOut[3] = {BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE};
    int shGrad[2]   = {BATCH_SIZE * SEQ_LEN, vocab.vocab_size};

    Tensor *X       = tensor_zeros(shX, 3);
    Tensor *Y       = tensor_zeros(shY, 2);
    Tensor *rnn_out = tensor_zeros(shRnnOut, 3);
    Tensor *pred    = tensor_zeros(shGrad, 2);
    Tensor *grad    = tensor_zeros(shGrad, 2);

    int sh_rnn_out_2d[2] = {BATCH_SIZE * SEQ_LEN, HIDDEN_SIZE};

    /* ── training loop ── */
    printf("Training...\n");
    for (int step = 0; step <= TRAIN_STEPS; step++) {
        make_batch(text, text_len, &vocab, X, Y);

        /* forward: RNN -> reshape -> Dense */
        rnn_forward(rnn, X, /*h_init=*/NULL, rnn_out);
        Tensor *rnn_out_2d = tensor_reshape(rnn_out, sh_rnn_out_2d, 2);
        dense_forward(out_layer, rnn_out_2d, pred);
        /* NOTE: rnn_out_2d must NOT be freed here. dense_forward()
         * caches the raw input pointer as l->input (see layer.c:
         * "cache input pointer (not owned)") for dense_backward() to
         * read later in this same step — freeing rnn_out_2d before
         * that read happens is a use-after-free (confirmed by ASan:
         * heap-use-after-free at layer.c:207 inside dense_backward,
         * on memory freed one line too early, right here). Freed
         * below instead, after dense_backward has already read it. */

        /* loss + initial gradient */
        float loss = nn_loss(LOSS_CROSS_ENTROPY, pred, Y, grad);

        /* backward: Dense -> reshape dX -> RNN
         * (rnn_zero_grad() called defensively before rnn_backward():
         * rnn_backward's doc says it "fills" l->dW_xh/dW_hh/db_h, but
         * dense_backward's equivalent zeroes internally per layer.c —
         * zeroing explicitly here is cheap insurance either way.) */
        dense_backward(out_layer, grad);
        /*
         * WORKAROUND for a library bug in layer.c/nn.c, not fixed here
         * on purpose (local file only, see conversation) — nn_loss()'s
         * LOSS_CROSS_ENTROPY case already divides its returned grad by
         * batch (nn.c:213), and dense_backward()'s ACT_SOFTMAX path
         * copies that grad straight through as dz with no further
         * scaling (layer.c:225-229) — but dense_backward then divides
         * dW/db by batch AGAIN internally (layer.c:279 and :291). Net
         * effect: dW/db end up scaled by 1/batch^2 instead of 1/batch,
         * making every weight update ~batch times too small (batch=400
         * here, i.e. ~400x) — this is exactly why loss was stuck flat
         * at ln(vocab) before this fix, not moving at all.
         * dX (layer.c:294-306) is NOT affected — dense_backward applies
         * no extra scaling there, so only dW/db need correcting; scaling
         * dX too would instead make it batch times too LARGE and feed a
         * wrongly-scaled gradient into rnn_backward below.
         */
        {
            float batch_correction = (float)(BATCH_SIZE * SEQ_LEN);
            for (size_t i = 0; i < out_layer->dW->size; i++)
                out_layer->dW->data[i] *= batch_correction;
            for (size_t i = 0; i < out_layer->db->size; i++)
                out_layer->db->data[i] *= batch_correction;
        }
        tensor_free(rnn_out_2d);   /* safe now — dense_backward's last read of it already happened */
        Tensor *dX_3d = tensor_reshape(out_layer->dX, shRnnOut, 3);
        rnn_zero_grad(rnn);
        rnn_backward(rnn, dX_3d);
        tensor_free(dX_3d);

        /* gradient clipping + SGD update */
        rnn_clip_gradients(rnn, GRAD_CLIP_NORM);
        rnn_update_sgd(rnn, LEARNING_RATE);
        dense_sgd_update(out_layer, LEARNING_RATE);

        if (step % PRINT_EVERY == 0) {
            printf("\n  Step %4d  loss=%.4f\n  Sample: ", step, loss);
            generate_sample(rnn, out_layer, &vocab, text[0], SAMPLE_LEN);
        }
    }

    printf("\nDone.\n\n");

    /* ── save trained weights for pipeline.c to load ─────────────── */
    if (save_model("model.bin", rnn, out_layer, vocab.vocab_size, HIDDEN_SIZE) == 0) {
        printf("Saved trained weights to model.bin (vocab_size=%d, hidden_size=%d)\n",
               vocab.vocab_size, HIDDEN_SIZE);
    } else {
        fprintf(stderr, "warning: failed to save model.bin\n");
    }

    tensor_free(X); tensor_free(Y);
    tensor_free(rnn_out); tensor_free(pred); tensor_free(grad);
    rnn_free(rnn);
    dense_free(out_layer);
    free(text);

    return 0;
}
