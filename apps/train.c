/*
 * train.c — trains the char/word-level RNN language model used by
 * pipeline.c for inference.
 *
 * Pipeline (mirrors pipeline.c's own header style):
 *   1. tokenizer_load(vocab.txt)                          — tokenizer.h
 *   2. read data.txt whole, tokenizer_encode() it once     — tokenizer.h
 *   3. write that token stream out as dataset_loader.h's
 *      binary format, then dataset_open() it               — dataset_loader.h
 *   4. rnn_create()/dense_create() the model                — rnn.h/layer.h
 *   5. per step: dataset_next_batch() -> one-hot ->
 *      rnn_forward -> dense_forward -> nn_loss(CROSS_ENTROPY)
 *      -> dense_backward -> rnn_backward -> clip -> sgd_step  — nn.h/optimizer.h
 *   6. save a checkpoint in the SAME layout demo_char_rnn.c's
 *      save_model()/pipeline.c's model_load() already use.
 *
 * ── Two things worth knowing up front, both load-bearing ───────────
 *
 * (a) WHY THIS DOESN'T USE nn_forward()/nn_backward():
 *     nn_forward()'s generic per-layer dispatch hard-requires Dense
 *     input to be exactly 2D (CF_CHECK(cur->ndim == 2)), but RNN
 *     output is always 3D ([batch, seq, hidden]) — nn_forward() never
 *     reshapes between layers, so registering [RNN, Dense] on one
 *     Network and calling nn_forward() on it would abort at runtime
 *     the moment training starts. This file drives rnn_forward()/
 *     dense_forward() directly instead (same approach demo_char_rnn.c
 *     itself already uses), bridging the RNN(3D)<->Dense(2D) gap with
 *     tensor_reshape() — which is safe and cheap: it's documented as
 *     a view over the SAME underlying buffer (row-major [batch,seq,H]
 *     and [batch*seq,H] are the same bytes), not a copy.
 *     A Network* IS still built (nn_add_rnn + nn_add_dense) — not to
 *     drive forward/backward, only so sgd_step() can walk it and
 *     apply the right per-layer-type update (momentum-SGD for Dense,
 *     rnn_update_sgd() for the RNN) to weights this file populated
 *     gradients into by hand. sgd_step() doesn't care how the
 *     gradients got there.
 *
 * (b) WHY THIS DOESN'T USE nn_save()/nn_save_model():
 *     That's a real, working serializer — but it's a DIFFERENT format
 *     (magic/version/per-layer-type-tagged) than the plain
 *     "int32 vocab_size, int32 hidden_size, then raw W_xh/W_hh/b_h/
 *     dense-W/dense-b floats" layout demo_char_rnn.c's save_model()
 *     writes and pipeline.c's model_load() reads. Saving via
 *     nn_save_model() here would produce a model.bin pipeline.c can't
 *     load at all. This file's own save_model() (below) matches
 *     pipeline.c's expected layout exactly, byte for byte.
 *
 * Build (adjust include/link paths to your tree):
 *   cc -O2 -Wall -Wextra -DUSE_OMP -fopenmp \
 *      tokenizer.c dataset_loader.c tensor.c layer.c rnn.c nn.c \
 *      optimizer.c train.c -o train -lm
 * Run:
 *   ./train data.txt vocab.txt model.bin 20
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef USE_OMP
#include <omp.h>
#endif

#include "tokenizer.h"
#include "dataset_loader.h"
#include "tensor.h"
#include "layer.h"
#include "rnn.h"
#include "nn.h"
#include "optimizer.h"

/* ── menuconfig-controlled hyperparameters ───────────────────────────
 * Every one of these is a plain #ifndef fallback: if neuralc_config.h
 * (or -D on the command line) already defines it, that value wins —
 * this file only supplies a default for a from-scratch build. */

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif
#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 256
#endif
#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 20
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef EPOCHS
#define EPOCHS 20
#endif
#ifndef TEMPERATURE
#define TEMPERATURE 0.8f
#endif

typedef struct {
    int seq_len;
    int batch_size;
    int epochs;
    int hidden_size;
    float learning_rate;
    float temperature;

    int auto_mode;
    int override_vocab;
    int custom_vocab_size;
} TrainConfig;

TrainConfig default_config(void) {
    TrainConfig config = {
        .seq_len = SEQ_LEN,
        .batch_size = BATCH_SIZE,
        .epochs = EPOCHS,
        .hidden_size = HIDDEN_SIZE,
        .learning_rate = LEARNING_RATE,
        .temperature = TEMPERATURE,
        .auto_mode = 0,
        .override_vocab = 0,
        .custom_vocab_size = 0
    };
    return config;
}

/* Not in the original macro list, but direct parameters of the exact
 * calls this file makes (sgd_create/rnn_clip_gradients) — following
 * the same "respect a menuconfig override, else default" pattern. */
#ifndef MOMENTUM
#define MOMENTUM 0.9f
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.0f
#endif
#ifndef GRAD_CLIP_NORM
#define GRAD_CLIP_NORM 5.0f
#endif

/* ═══════════════════════════════════════════════════════════════════
 *  Checkpoint I/O — MUST stay byte-for-byte identical to
 *  demo_char_rnn.c's save_model() / pipeline.c's model_load(). See
 *  note (b) above for why this isn't nn_save_model().
 * ═══════════════════════════════════════════════════════════════════ */
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
    ok = ok && (write_tensor(f, rnn->W_xh)     == 0);
    ok = ok && (write_tensor(f, rnn->W_hh)     == 0);
    ok = ok && (write_tensor(f, rnn->b_h)      == 0);
    ok = ok && (write_tensor(f, out_layer->W)  == 0);
    ok = ok && (write_tensor(f, out_layer->b)  == 0);

    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════
 *  data.txt -> tokens -> dataset_loader.h's binary format
 * ═══════════════════════════════════════════════════════════════════
 * dataset_loader.h expects an ALREADY-tokenized dataset.bin (see its
 * own header comment: "sequential, no seeking" over a fixed uint32_t
 * array) — it has no notion of raw text or a tokenizer at all. So
 * this step exists purely to bridge tokenizer.h's output into a form
 * dataset_open()/dataset_next_batch() can stream, reusing their
 * already-correct (and already tested) wraparound batching instead of
 * reimplementing it here.
 *
 * ── EOS handling ─────────────────────────────────────────────────
 * data.txt is one training example per line. If the loaded vocab
 * declares an EOS token (tokenizer_eos_id() != TOKENIZER_INVALID_ID),
 * each line is encoded SEPARATELY and that raw id is appended after
 * every line's tokens — inserted directly into the id stream, the
 * same way cli.c/sanity_test.c already consume eos_id as a bare
 * integer, never through text matching. This is deliberate: EOS is a
 * control token, not a natural-language word, so it should never
 * need to round-trip through tokenizer_encode()'s marker-based
 * word-matching to end up in the stream. Encoding line-by-line (vs.
 * one encode() over the whole file) is what makes this possible —
 * inserting a mid-stream id after the whole-file version's single
 * encode() call would have no natural boundary to insert *at*.
 *
 * If the vocab has no EOS entry, this falls back to the exact
 * previous behavior: one encode() over the whole file, no per-line
 * splitting, no ids inserted. Training still works — it just never
 * sees a stop signal, same as before this change. */
static int tokenize_corpus_to_dataset_bin(const char *data_path,
                                           const Tokenizer *tok,
                                           const char *out_bin_path,
                                           size_t *out_token_count) {
    FILE *f = fopen(data_path, "rb");
    if (!f) {
        fprintf(stderr, "train: cannot open corpus '%s'\n", data_path);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsize = ftell(f);
    if (fsize < 0) { fclose(f); return -1; }
    rewind(f);

    char *text = (char *)malloc((size_t)fsize + 1);
    if (!text) { fclose(f); return -1; }
    size_t rd = fread(text, 1, (size_t)fsize, f);
    fclose(f);
    text[rd] = '\0';

    int eos_id = tokenizer_eos_id(tok);

    int n;
    int *token_ids;

    if (eos_id == TOKENIZER_INVALID_ID) {
        /* No EOS declared: unchanged from before — one encode() over
         * the whole file, upper-bounded by byte count. */
        token_ids = (int *)malloc(rd > 0 ? rd * sizeof(int) : sizeof(int));
        if (!token_ids) { free(text); return -1; }
        n = tokenizer_encode(tok, text, token_ids, (int)(rd > 0 ? rd : 1));
    } else {
        /* Upper bound: at most `rd` real tokens (same bound as
         * above) plus one EOS id per line. Counting '\n' bytes bounds
         * the line count regardless of whether the file ends with a
         * trailing newline. */
        size_t line_count = 1;
        for (size_t i = 0; i < rd; i++)
            if (text[i] == '\n') line_count++;

        size_t cap = (rd > 0 ? rd : 1) + line_count;
        token_ids = (int *)malloc(cap * sizeof(int));
        if (!token_ids) { free(text); return -1; }

        n = 0;
        char *line_start = text;
        int reached_end = 0;
        while (!reached_end) {
            char *nl = strchr(line_start, '\n');
            if (nl) {
                *nl = '\0';
            } else {
                reached_end = 1;
            }

            /* strip a trailing \r (CRLF corpora) */
            size_t len = strlen(line_start);
            if (len > 0 && line_start[len - 1] == '\r') {
                line_start[len - 1] = '\0';
                len--;
            }

            if (len > 0) {
                int remaining = (int)(cap - (size_t)n);
                int line_n = tokenizer_encode(tok, line_start,
                                               token_ids + n, remaining);
                if (line_n > 0) {
                    n += line_n;
                    /* Room is guaranteed: cap reserved one extra slot
                     * per line up front. */
                    token_ids[n++] = eos_id;
                }
            }

            if (reached_end) break;
            line_start = nl + 1;
        }
    }

    free(text);
    if (n <= 0) {
        fprintf(stderr, "train: tokenization of '%s' produced no tokens\n", data_path);
        free(token_ids);
        return -1;
    }

    FILE *out = fopen(out_bin_path, "wb");
    if (!out) {
        fprintf(stderr, "train: cannot open '%s' for writing\n", out_bin_path);
        free(token_ids);
        return -1;
    }

    /* DatasetHeader comes from dataset_loader.h itself (not
     * reimplemented here), so field widths/layout are always
     * whatever that real header says — this file only assigns fields
     * by name, never assumes a byte layout. version must match
     * dataset_loader.c's own EXPECTED_VERSION (currently 1) — if that
     * ever changes, update it here too. */
    DatasetHeader header;
    memset(&header, 0, sizeof header);
    header.magic       = DATASET_MAGIC;
    header.version     = 1;
    header.token_count = (uint64_t)n;

    int ok = (fwrite(&header, sizeof header, 1, out) == 1);
    for (int i = 0; ok && i < n; i++) {
        uint32_t tid = (uint32_t)token_ids[i];
        ok = (fwrite(&tid, sizeof tid, 1, out) == 1);
    }
    free(token_ids);
    if (fclose(out) != 0) ok = 0;

    if (!ok) {
        fprintf(stderr, "train: failed writing dataset file '%s'\n", out_bin_path);
        return -1;
    }

    *out_token_count = (size_t)n;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Batch -> one-hot tensors
 * ═══════════════════════════════════════════════════════════════════
 * dataset_next_batch() hands back raw uint32_t token ids; rnn_forward
 * needs a one-hot [batch,seq,vocab] float input, and nn_loss's
 * LOSS_CROSS_ENTROPY needs a one-hot [batch*seq,vocab] float target
 * (see nn.c: it reads target as a float multiplier t*log(p), i.e. a
 * genuine one-hot row, not a bare class index). Each (b,t) writes to
 * disjoint memory — safe to parallelize across threads. */
static void fill_onehot_3d(Tensor *dst, const uint32_t *ids,
                           int batch, int seq, int vocab_size) {
    tensor_fill(dst, 0.0f);
#ifdef USE_OMP
    #pragma omp parallel for collapse(2) schedule(static) if(batch*seq > 64)
#endif
    for (int b = 0; b < batch; b++)
        for (int t = 0; t < seq; t++)
            dst->data[(size_t)b*seq*vocab_size + (size_t)t*vocab_size
                      + ids[b*seq + t]] = 1.0f;
}

static void fill_onehot_2d(Tensor *dst, const uint32_t *ids,
                           int rows, int vocab_size) {
    tensor_fill(dst, 0.0f);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(rows > 64)
#endif
    for (int r = 0; r < rows; r++)
        dst->data[(size_t)r*vocab_size + ids[r]] = 1.0f;
}

/* ═══════════════════════════════════════════════════════════════════
 *  main
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <data.txt> <vocab.txt> <model.bin> [epochs] [seq_len] [batch_size] [learning_rate] [auto_mode] [custom_vocab_size]\n",
                argv[0]);
        return 1;
    }
    const char *data_path  = argv[1];
    const char *vocab_path = argv[2];
    const char *model_path = argv[3];
    TrainConfig config = default_config();
    if (argc > 4) config.epochs = atoi(argv[4]);
    if (argc > 5) config.seq_len = atoi(argv[5]);
    if (argc > 6) config.batch_size = atoi(argv[6]);
    if (argc > 7) config.learning_rate = strtof(argv[7], NULL);
    if (argc > 8) config.auto_mode = (atoi(argv[8]) == 1);
    if (argc > 9) {
        config.custom_vocab_size = atoi(argv[9]);
        if (config.custom_vocab_size > 0) config.override_vocab = 1;
    }
    if (config.epochs <= 0) config.epochs = default_config().epochs;
    if (config.seq_len <= 0) config.seq_len = default_config().seq_len;
    if (config.batch_size <= 0) config.batch_size = default_config().batch_size;
    if (config.learning_rate <= 0.0f)
        config.learning_rate = default_config().learning_rate;

    printf("[config] epochs=%d seq_len=%d batch=%d lr=%f hidden=%d auto=%d vocab_override=%d\n",
           config.epochs, config.seq_len, config.batch_size,
           config.learning_rate, config.hidden_size, config.auto_mode,
           config.override_vocab);

    printf("[train] Loading tokenizer from '%s'...\n", vocab_path);
    Tokenizer *tok = tokenizer_load(vocab_path);
    if (!tok) { fprintf(stderr, "train: failed to load tokenizer\n"); return 1; }
    int vocab_size = tokenizer_vocab_size(tok);
    if (config.override_vocab && config.custom_vocab_size > 0) {
        if (config.custom_vocab_size < vocab_size) {
            printf("[config] Warning: custom vocab_size %d is smaller than tokenizer vocab %d; using tokenizer vocab to keep token IDs valid\n",
                   config.custom_vocab_size, vocab_size);
        } else {
            printf("[config] Overriding vocab_size: %d -> %d\n",
                   vocab_size, config.custom_vocab_size);
            vocab_size = config.custom_vocab_size;
        }
    }
    printf("[train] vocab_size=%d\n", vocab_size);

    printf("[train] Tokenizing '%s'...\n", data_path);
    char dataset_bin_path[1024];
    snprintf(dataset_bin_path, sizeof dataset_bin_path, "%s.dataset.bin", data_path);
    size_t token_count = 0;
    if (tokenize_corpus_to_dataset_bin(data_path, tok, dataset_bin_path, &token_count) != 0) {
        tokenizer_free(tok);
        return 1;
    }
    printf("[train] %zu tokens written to '%s'\n", token_count, dataset_bin_path);

    printf("[train] Loading dataset...\n");
    Dataset *ds = dataset_open(dataset_bin_path);
    if (!ds) { fprintf(stderr, "train: failed to open '%s'\n", dataset_bin_path); tokenizer_free(tok); return 1; }

    if (ds->token_count < (size_t)(config.seq_len + 1)) {
        fprintf(stderr,
            "train: corpus has only %zu tokens, need at least SEQ_LEN+1=%d — "
            "use a longer data.txt or a smaller SEQ_LEN\n",
            ds->token_count, config.seq_len + 1);
        dataset_close(ds);
        tokenizer_free(tok);
        return 1;
    }


    RNNLayer *rnn = NULL;
    DenseLayer *dense = NULL;

    if (access(model_path, F_OK) == 0) {
        printf("[train] Loading existing model from '%s' ...\n", model_path);

        FILE *f = fopen(model_path, "rb");
        if (!f) {
            fprintf(stderr, "[train] Failed to open model file \n");
            return 1;
        }

        int32_t vs = 0, hs = 0;
        if (fread(&vs, sizeof(int32_t), 1, f) != 1 ||
            fread(&hs, sizeof(int32_t), 1, f) != 1) {
            fprintf(stderr, "[train] Error reading model header\n");
            fclose(f);
            return 1;
        }

        if (vs != vocab_size) {
            printf("[train] Warning: vocab mismatch (%d vs %d)\n", vs, vocab_size);

        }

        rnn = rnn_create(vocab_size, hs);
        dense = dense_create(hs, vocab_size, ACT_SOFTMAX);

        size_t n1 = fread(rnn->W_xh->data, sizeof(float), rnn->W_xh->size, f);
        size_t n2 = fread(rnn->W_hh->data, sizeof(float), rnn->W_hh->size, f);
        size_t n3 = fread(rnn->b_h->data,  sizeof(float), rnn->b_h->size,  f);
        size_t n4 = fread(dense->W->data,  sizeof(float), dense->W->size,  f);
        size_t n5 = fread(dense->b->data,  sizeof(float), dense->b->size,  f);
        (void)n1; (void)n2; (void)n3; (void)n4; (void)n5;
    
        fclose(f);

        printf("[train] MODEL loaded. Continuing training ...\n");

    } else {
        printf("[train] Creating new model (hidden_size=%d)...\n", config.hidden_size);

        rnn = rnn_create(vocab_size, config.hidden_size);
        dense = dense_create(config.hidden_size, vocab_size, ACT_SOFTMAX);
         
    
    }

    //  the above line can save the funtions in load all the model.bin files

    /*
    printf("[train] Creating model (hidden_size=%d)...\n", config.hidden_size);
    RNNLayer   *rnn = rnn_create(vocab_size, config.hidden_size);
    // ACT_SOFTMAX, not ACT_NONE: nn_loss(LOSS_CROSS_ENTROPY,...)'s
     * combined (p-t) gradient (see nn.c) assumes `pred` is already a
     * genuine softmax probability, and dense_backward()'s ACT_SOFTMAX
     * case correspondingly passes that gradient straight through as
     * dz without re-differentiating softmax. Both halves only agree
     * with each other when this layer is ACT_SOFTMAX. (Contrast with
     * pipeline.c, which deliberately uses ACT_NONE at INFERENCE time
     * to get raw logits for temperature/top-p sampling — a different
     * stage with a different requirement.) */
   /*DenseLayer *dense = dense_create(config.hidden_size, vocab_size, ACT_SOFTMAX);*/

    /* Built only so sgd_step() can dispatch per-layer-type updates to
     * these same rnn/dense pointers — see note (a) at the top of this
     * file. Forward/backward below never goes through this Network. */
    Network *net = nn_create();
    nn_add_rnn(net, rnn);
    nn_add_dense(net, dense);

    printf("[train] Creating optimizer (lr=%g, momentum=%g, weight_decay=%g)...\n",
           (double)config.learning_rate, (double)MOMENTUM, (double)WEIGHT_DECAY);
    SGD *opt = sgd_create(config.learning_rate, MOMENTUM, WEIGHT_DECAY);

    /* Reused every step — sized once, never resized (batch_size/seq_len
     * are fixed for the whole run), avoiding per-step malloc/free. */
    uint32_t *input_ids  = (uint32_t *)malloc((size_t)config.batch_size * config.seq_len * sizeof(uint32_t));
    uint32_t *target_ids = (uint32_t *)malloc((size_t)config.batch_size * config.seq_len * sizeof(uint32_t));

    int sh_in3[3]  = { config.batch_size, config.seq_len, vocab_size };
    int sh_out3[3] = { config.batch_size, config.seq_len, config.hidden_size };
    int sh_2d_h[2] = { config.batch_size * config.seq_len, config.hidden_size };
    int sh_2d_v[2] = { config.batch_size * config.seq_len, vocab_size };

    Tensor *input3d  = tensor_zeros(sh_in3,  3);   /* one-hot RNN input           */
    Tensor *rnn_out3d = tensor_zeros(sh_out3, 3);   /* RNN hidden states, per t    */
    Tensor *dense_out = tensor_zeros(sh_2d_v, 2);   /* softmax probs, flattened    */
    Tensor *target2d  = tensor_zeros(sh_2d_v, 2);   /* one-hot targets, flattened  */
    Tensor *loss_grad = tensor_zeros(sh_2d_v, 2);   /* dL/d(dense_out)             */

    if (!input_ids || !target_ids || !input3d || !rnn_out3d ||
        !dense_out || !target2d || !loss_grad) {
        fprintf(stderr, "train: out of memory allocating training buffers\n");
        goto cleanup;
    }

    /* One epoch = a fixed number of batches derived from corpus size.
     * dataset_next_batch() itself has no notion of an "epoch boundary"
     * (it streams indefinitely via wraparound — see dataset_loader.c's
     * own header comment) — this file layers "epoch" on top purely by
     * counting a fixed number of steps per epoch. */
    {
        size_t tokens_per_batch = (size_t)config.batch_size * config.seq_len;
        int steps_per_epoch = (int)(ds->token_count / tokens_per_batch);
        if (steps_per_epoch < 1) steps_per_epoch = 1;

        printf("[train] %d steps/epoch, %d epochs, batch_size=%d, seq_len=%d\n",
               steps_per_epoch, config.epochs, config.batch_size, config.seq_len);

        float prev_loss = 1e9f;
        int plateau_count = 0;

        for (int epoch = 0; epoch < config.epochs; epoch++) {
            double total_loss = 0.0;

            for (int step = 0; step < steps_per_epoch; step++) {
                if (dataset_next_batch(ds, input_ids, target_ids,
                                       config.batch_size, config.seq_len) != 0) {
                    fprintf(stderr, "train: dataset_next_batch failed at "
                                    "epoch %d step %d\n", epoch, step);
                    goto cleanup;
                }

                fill_onehot_3d(input3d, input_ids, config.batch_size, config.seq_len, vocab_size);
                fill_onehot_2d(target2d, target_ids,
                                config.batch_size * config.seq_len, vocab_size);

                /* forward: RNN over the whole sequence, then Dense
                 * per-timestep via a reshape view (no copy — see note
                 * (a) at the top of this file). rnn_out_2d must stay
                 * alive until AFTER dense_backward() below: DenseLayer
                 * caches the pointer it was called with in l->input
                 * (see layer.c's dense_forward — "cache input pointer
                 * (not owned)") and reads it again during
                 * dense_backward()'s dW computation. Freeing this view
                 * right after dense_forward() leaves that a dangling
                 * pointer — confirmed via ASan during testing, this is
                 * not a hypothetical. */
                rnn_forward(rnn, input3d, /*h_init=*/NULL, rnn_out3d);
                Tensor *rnn_out_2d = tensor_reshape(rnn_out3d, sh_2d_h, 2);
                dense_forward(dense, rnn_out_2d, dense_out);

                float loss = nn_loss(LOSS_CROSS_ENTROPY, dense_out, target2d, loss_grad);
                total_loss += loss;

                /* backward: Dense first (still using rnn_out_2d via
                 * its cached l->input), THEN free that view, then
                 * reshape dense->dX back to 3D for the RNN's own BPTT */
                dense_backward(dense, loss_grad);
                tensor_free(rnn_out_2d);   /* safe now — dense_backward() is done reading it */
                Tensor *dX_3d = tensor_reshape(dense->dX, sh_out3, 3);
                rnn_backward(rnn, dX_3d);
                tensor_free(dX_3d);   /* view only */

                /* clip BOTH layers' gradients — nn_clip_gradients()
                 * (optimizer.c) only ever touches Dense's dW/db (see
                 * its own header comment), so the RNN needs its own
                 * clip call too, or exploding BPTT gradients on the
                 * recurrent weights would go completely unchecked. */
                rnn_clip_gradients(rnn, GRAD_CLIP_NORM);
                nn_clip_gradients(net, GRAD_CLIP_NORM);

                sgd_step(opt, net);
            }

            float avg_loss = (float)(total_loss / steps_per_epoch);

            if (config.auto_mode) {
                if (avg_loss > prev_loss - 0.0005f) {
                    plateau_count++;
                } else {
                    plateau_count = 0;
                }

                if (plateau_count >= 3) {
                    config.learning_rate *= 0.8f;
                    opt->lr = config.learning_rate;
                    plateau_count = 0;
                    printf("[auto] Plateau detected → reducing lr to %f\n",
                           config.learning_rate);
                }
            }

            prev_loss = avg_loss;
            printf("Epoch %d | Loss: %.4f\n", epoch, avg_loss);
        }
    }

    printf("[train] Saving model to '%s'...\n", model_path);
    if (save_model(model_path, rnn, dense, vocab_size, config.hidden_size) != 0) {
        fprintf(stderr, "train: failed to save model\n");
    } else {
        printf("[train] Done.\n");
    }

cleanup:
    free(input_ids);
    free(target_ids);
    tensor_free(input3d);
    tensor_free(rnn_out3d);
    tensor_free(dense_out);
    tensor_free(target2d);
    tensor_free(loss_grad);
    sgd_free(opt);
    nn_free(net);   /* frees rnn and dense too — they're registered on it */
    dataset_close(ds);
    tokenizer_free(tok);
    return 0;
}
