/*
 * load_model.c — loads a serialized RNN/LSTM + Dense inference model
 * from a custom binary (.bin) format into memory.
 *
 * This module does exactly one job: turn a trusted-format-but-
 * untrusted-CONTENT file on disk into a fully populated, validated
 * Model* (or NULL on any failure, with everything already cleaned
 * up). No training, no tokenizer, no dynamic resizing, no file
 * writing, no threading, no dependencies beyond the C standard
 * library.
 *
 * ── On-disk layout (strict, sequential, no seeking) ────────────────
 *
 *   ModelHeader                              (fixed size)
 *   if model_type == 0 (RNN):
 *     RNNLayerHeader
 *     float Wx[input_size * hidden_size]
 *     float Wh[hidden_size * hidden_size]
 *     float b [hidden_size]
 *   if model_type == 1 (LSTM):
 *     LSTMLayerHeader
 *     float Wx[4 * input_size  * hidden_size]   gate order [i,f,g,o]
 *     float Wh[4 * hidden_size * hidden_size]
 *     float b [4 * hidden_size]
 *   DenseLayerHeader
 *   float W[in_features * out_features]
 *   float b[out_features]
 *
 * All multi-byte header fields are read as-is (assumes a
 * little-endian host, IEEE-754 float32 — as specified). All structs
 * are byte-packed so sizeof() matches the on-disk layout exactly.
 *
 * ── Safety posture ──────────────────────────────────────────────────
 * Every dimension used to size an allocation comes straight from the
 * file, i.e. is attacker-controlled. Two things are done about that
 * which the spec doesn't spell out line-by-line but "crash-safe" and
 * "production-ready" both require:
 *   1. Every size computation (e.g. input_size * hidden_size) is done
 *      in size_t with an explicit overflow check before it's ever
 *      passed to malloc/fread — a crafted header can't wrap a huge
 *      dimension around into a tiny allocation followed by a huge
 *      fread.
 *   2. A generous but finite per-array element cap
 *      (MAX_ARRAY_ELEMENTS) rejects absurd dimensions outright with a
 *      clear error, instead of attempting a multi-gigabyte malloc.
 * Neither changes the documented binary format or any of the
 * mandatory validation rules — they only harden the arithmetic those
 * rules are built on.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "load_model.h"

/* ═══════════════════════════════════════════════════════════════════
 *  On-disk structs (packed — must match the binary format exactly)
 * ═══════════════════════════════════════════════════════════════════ */

#pragma pack(push, 1)

typedef struct {
    char     magic[4];        /* must be "NCRN"        */
    uint32_t version;         /* must be 1              */
    uint32_t model_type;      /* 0 = RNN, 1 = LSTM       */
    uint32_t vocab_size;      /* must be 256             */
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t reserved[8];
} ModelHeader;

typedef struct {
    uint32_t input_size;
    uint32_t hidden_size;
} RNNLayerHeader;

typedef struct {
    uint32_t input_size;
    uint32_t hidden_size;
} LSTMLayerHeader;

typedef struct {
    uint32_t in_features;
    uint32_t out_features;
} DenseLayerHeader;

#pragma pack(pop)

/* ═══════════════════════════════════════════════════════════════════
 *  Constants
 * ═══════════════════════════════════════════════════════════════════ */

#define EXPECTED_MAGIC   "NCRN"
#define EXPECTED_VERSION 1u
#define EXPECTED_VOCAB   256u

#define MODEL_TYPE_RNN   0u
#define MODEL_TYPE_LSTM  1u

/* Hard ceiling on any single weight array's element count — guards
 * the overflow-safe multiplications below against a file that claims
 * an absurd hidden_size/input_size and would otherwise try to
 * malloc()/fread() many gigabytes. 256M floats = 1GB, comfortably
 * above anything a real char-level RNN/LSTM checkpoint needs. */
#define MAX_ARRAY_ELEMENTS ((size_t)256u * 1024u * 1024u)

/* ═══════════════════════════════════════════════════════════════════
 *  Small internal helpers
 * ═══════════════════════════════════════════════════════════════════ */

/* Wraps fread() with return-value validation, per the "no partial
 * reads go unnoticed" rule. `what` names the field being read, for a
 * clear error message. Returns 1 on success, 0 on short read/EOF/
 * error (message already printed). */
static int read_exact(FILE *f, void *buf, size_t nmemb, size_t size,
                      const char *what) {
    size_t got = fread(buf, size, nmemb, f);
    if (got != nmemb) {
        fprintf(stderr, "Error: failed to read %s (expected %zu elements, got %zu)\n",
                what, nmemb, got);
        return 0;
    }
    return 1;
}

/* Overflow-safe size_t multiplication with an upper bound. Returns 1
 * and writes the product to *out on success; returns 0 (no write) if
 * the multiplication would overflow size_t or exceed
 * MAX_ARRAY_ELEMENTS. */
static int safe_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return 0;   /* would overflow */
    size_t product = a * b;
    if (product > MAX_ARRAY_ELEMENTS) return 0;  /* absurd for this format */
    *out = product;
    return 1;
}

/* malloc's `count` floats, validated via safe_mul. Returns NULL on
 * overflow, out-of-range size, or malloc failure — caller treats NULL
 * as "load failed", triggering full cleanup. `count == 0` is treated
 * as a valid empty allocation request and also returns NULL (there is
 * no legitimate zero-sized weight array in this format, so any such
 * case is itself an error the caller should reject). */
static float *alloc_floats(size_t count, const char *what) {
    if (count == 0 || count > MAX_ARRAY_ELEMENTS) {
        fprintf(stderr, "Error: invalid size for %s (%zu elements)\n", what, count);
        return NULL;
    }
    float *buf = (float *)malloc(count * sizeof(float));
    if (!buf) {
        fprintf(stderr, "Error: out of memory allocating %s (%zu floats)\n",
                what, count);
        return NULL;
    }
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════
 *  free_model — safe under NULL, partial init, or full init
 * ═══════════════════════════════════════════════════════════════════ */

void free_model(Model *m) {
    if (!m) return;

    free(m->Wx);
    free(m->Wh);
    free(m->b);

    free(m->Wx_lstm);
    free(m->Wh_lstm);
    free(m->b_lstm);

    free(m->W_dense);
    free(m->b_dense);

    free(m);
}

/* ═══════════════════════════════════════════════════════════════════
 *  load_model
 * ═══════════════════════════════════════════════════════════════════ */

Model *load_model(const char *path) {
    if (!path) {
        fprintf(stderr, "Error: null path passed to load_model\n");
        return NULL;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open model file '%s'\n", path);
        return NULL;
    }

    /* Model is calloc'd, so every pointer starts NULL — free_model()
     * is safe to call at any point below, no matter how far we got. */
    Model *m = (Model *)calloc(1, sizeof(Model));
    if (!m) {
        fprintf(stderr, "Error: failed to allocate Model struct\n");
        fclose(f);
        return NULL;
    }

    /* ── header ──────────────────────────────────────────────────── */
    ModelHeader header;
    if (!read_exact(f, &header, 1, sizeof(header), "ModelHeader")) {
        goto fail;
    }

    if (memcmp(header.magic, EXPECTED_MAGIC, 4) != 0) {
        fprintf(stderr, "Error: invalid magic (expected \"NCRN\")\n");
        goto fail;
    }
    if (header.version != EXPECTED_VERSION) {
        fprintf(stderr, "Error: unsupported version %u (expected %u)\n",
                header.version, EXPECTED_VERSION);
        goto fail;
    }
    if (header.vocab_size != EXPECTED_VOCAB) {
        fprintf(stderr, "Error: unsupported vocab_size %u (expected %u)\n",
                header.vocab_size, EXPECTED_VOCAB);
        goto fail;
    }
    if (header.model_type != MODEL_TYPE_RNN && header.model_type != MODEL_TYPE_LSTM) {
        fprintf(stderr, "Error: unsupported model_type %u (expected 0 or 1)\n",
                header.model_type);
        goto fail;
    }

    m->model_type  = header.model_type;
    m->vocab_size  = header.vocab_size;
    m->hidden_size = header.hidden_size;

    /* ── recurrent layer ─────────────────────────────────────────── */
    if (header.model_type == MODEL_TYPE_RNN) {
        RNNLayerHeader rh;
        if (!read_exact(f, &rh, 1, sizeof(rh), "RNNLayerHeader")) goto fail;

        if (rh.input_size != header.vocab_size) {
            fprintf(stderr,
                "Error: RNN input_size (%u) != vocab_size (%u)\n",
                rh.input_size, header.vocab_size);
            goto fail;
        }
        if (rh.hidden_size != header.hidden_size) {
            fprintf(stderr,
                "Error: RNN hidden_size (%u) != header hidden_size (%u)\n",
                rh.hidden_size, header.hidden_size);
            goto fail;
        }

        size_t n_wx, n_wh, n_b;
        if (!safe_mul(rh.input_size, rh.hidden_size, &n_wx)) {
            fprintf(stderr, "Error: RNN Wx dimensions too large or invalid\n");
            goto fail;
        }
        if (!safe_mul(rh.hidden_size, rh.hidden_size, &n_wh)) {
            fprintf(stderr, "Error: RNN Wh dimensions too large or invalid\n");
            goto fail;
        }
        n_b = rh.hidden_size;

        m->Wx = alloc_floats(n_wx, "RNN Wx");
        if (!m->Wx) goto fail;
        m->Wh = alloc_floats(n_wh, "RNN Wh");
        if (!m->Wh) goto fail;
        m->b = alloc_floats(n_b, "RNN b");
        if (!m->b) goto fail;

        if (!read_exact(f, m->Wx, n_wx, sizeof(float), "RNN Wx")) goto fail;
        if (!read_exact(f, m->Wh, n_wh, sizeof(float), "RNN Wh")) goto fail;
        if (!read_exact(f, m->b,  n_b,  sizeof(float), "RNN b"))  goto fail;

    } else { /* MODEL_TYPE_LSTM */
        LSTMLayerHeader lh;
        if (!read_exact(f, &lh, 1, sizeof(lh), "LSTMLayerHeader")) goto fail;

        if (lh.input_size != header.vocab_size) {
            fprintf(stderr,
                "Error: LSTM input_size (%u) != vocab_size (%u)\n",
                lh.input_size, header.vocab_size);
            goto fail;
        }
        if (lh.hidden_size != header.hidden_size) {
            fprintf(stderr,
                "Error: LSTM hidden_size (%u) != header hidden_size (%u)\n",
                lh.hidden_size, header.hidden_size);
            goto fail;
        }

        size_t n_wx, n_wh, n_b, tmp;
        /* 4 * input_size * hidden_size, computed as two checked
         * multiplications so an overflow at either step is caught. */
        if (!safe_mul(lh.input_size, lh.hidden_size, &tmp) ||
            !safe_mul(tmp, 4, &n_wx)) {
            fprintf(stderr, "Error: LSTM Wx dimensions too large or invalid\n");
            goto fail;
        }
        if (!safe_mul(lh.hidden_size, lh.hidden_size, &tmp) ||
            !safe_mul(tmp, 4, &n_wh)) {
            fprintf(stderr, "Error: LSTM Wh dimensions too large or invalid\n");
            goto fail;
        }
        if (!safe_mul(lh.hidden_size, 4, &n_b)) {
            fprintf(stderr, "Error: LSTM b dimensions too large or invalid\n");
            goto fail;
        }

        m->Wx_lstm = alloc_floats(n_wx, "LSTM Wx");
        if (!m->Wx_lstm) goto fail;
        m->Wh_lstm = alloc_floats(n_wh, "LSTM Wh");
        if (!m->Wh_lstm) goto fail;
        m->b_lstm = alloc_floats(n_b, "LSTM b");
        if (!m->b_lstm) goto fail;

        if (!read_exact(f, m->Wx_lstm, n_wx, sizeof(float), "LSTM Wx")) goto fail;
        if (!read_exact(f, m->Wh_lstm, n_wh, sizeof(float), "LSTM Wh")) goto fail;
        if (!read_exact(f, m->b_lstm,  n_b,  sizeof(float), "LSTM b"))  goto fail;
    }

    /* ── dense output layer ──────────────────────────────────────── */
    DenseLayerHeader dh;
    if (!read_exact(f, &dh, 1, sizeof(dh), "DenseLayerHeader")) goto fail;

    if (dh.in_features != header.hidden_size) {
        fprintf(stderr,
            "Error: Dense in_features (%u) != hidden_size (%u)\n",
            dh.in_features, header.hidden_size);
        goto fail;
    }
    if (dh.out_features != header.vocab_size) {
        fprintf(stderr,
            "Error: Dense out_features (%u) != vocab_size (%u)\n",
            dh.out_features, header.vocab_size);
        goto fail;
    }

    size_t n_w, n_db;
    if (!safe_mul(dh.in_features, dh.out_features, &n_w)) {
        fprintf(stderr, "Error: Dense W dimensions too large or invalid\n");
        goto fail;
    }
    n_db = dh.out_features;

    m->W_dense = alloc_floats(n_w, "Dense W");
    if (!m->W_dense) goto fail;
    m->b_dense = alloc_floats(n_db, "Dense b");
    if (!m->b_dense) goto fail;

    if (!read_exact(f, m->W_dense, n_w,  sizeof(float), "Dense W")) goto fail;
    if (!read_exact(f, m->b_dense, n_db, sizeof(float), "Dense b")) goto fail;

    fclose(f);
    return m;

fail:
    fclose(f);
    free_model(m);
    return NULL;
}
