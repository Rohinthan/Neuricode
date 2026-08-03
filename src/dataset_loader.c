/*
 * dataset_loader.c — sequential loader for the fixed binary token
 * dataset format (dataset.bin).
 *
 * ── On-disk layout (strict, sequential, no seeking) ────────────────
 *
 *   DatasetHeader                       (fixed size, packed)
 *   uint32_t tokens[token_count]         (raw, contiguous, no padding)
 *
 * Everything is read in one linear pass in dataset_open(): the whole
 * token array is loaded into memory once, so dataset_next_batch() is
 * pure memcpy over an in-memory buffer — no file I/O, no allocation,
 * on the hot path.
 *
 * ── Safety posture ──────────────────────────────────────────────────
 * token_count comes straight from the file and is therefore
 * attacker-/corruption-controlled. Before it's ever used to size an
 * allocation it is checked against SIZE_MAX / sizeof(uint32_t), so a
 * huge token_count fails cleanly instead of wrapping into a small
 * allocation followed by a huge fread. The same "never trust file
 * contents" posture from the model loader applies here: magic,
 * version, and reserved fields are all validated, an empty dataset is
 * rejected, and any trailing bytes after the expected token array are
 * treated as a corrupt file rather than silently ignored.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "dataset_loader.h"

#define EXPECTED_VERSION 1u

/* ═══════════════════════════════════════════════════════════════════
 *  Small internal helpers
 * ═══════════════════════════════════════════════════════════════════ */

/* Wraps fread() with return-value validation. `what` names the field
 * being read, for a clear error message. Returns 1 on success, 0 on
 * short read/EOF/error (message already printed). */
static int read_exact(FILE *f, void *buf, size_t nmemb, size_t size,
                      const char *what) {
    size_t got = fread(buf, size, nmemb, f);
    if (got != nmemb) {
        fprintf(stderr,
            "Error: failed to read %s (expected %zu elements, got %zu)\n",
            what, nmemb, got);
        return 0;
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 *  dataset_close — safe under NULL or partial init
 * ═══════════════════════════════════════════════════════════════════ */

void dataset_close(Dataset *ds) {
    if (!ds) return;
    if (ds->f) fclose(ds->f);
    free(ds->tokens);
    free(ds);
}

/* ═══════════════════════════════════════════════════════════════════
 *  dataset_open
 * ═══════════════════════════════════════════════════════════════════ */

Dataset *dataset_open(const char *path) {
    if (!path) {
        fprintf(stderr, "Error: null path passed to dataset_open\n");
        return NULL;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open dataset file '%s'\n", path);
        return NULL;
    }

    /* calloc'd so every field starts zero/NULL — dataset_close() is
     * safe to call at any point below, no matter how far we got. */
    Dataset *ds = (Dataset *)calloc(1, sizeof(Dataset));
    if (!ds) {
        fprintf(stderr, "Error: failed to allocate Dataset struct\n");
        fclose(f);
        return NULL;
    }
    ds->f = f;

    /* ── header ──────────────────────────────────────────────────── */
    DatasetHeader header;
    if (!read_exact(f, &header, 1, sizeof(header), "DatasetHeader"))
        goto fail;

    if (header.magic != DATASET_MAGIC) {
        fprintf(stderr, "Error: invalid magic (expected 0x%08X, got 0x%08X)\n",
                DATASET_MAGIC, header.magic);
        goto fail;
    }
    if (header.version != EXPECTED_VERSION) {
        fprintf(stderr, "Error: unsupported version %u (expected %u)\n",
                header.version, EXPECTED_VERSION);
        goto fail;
    }
    for (int i = 0; i < 5; i++) {
        if (header.reserved[i] != 0) {
            fprintf(stderr, "Error: reserved fields must be zero\n");
            goto fail;
        }
    }
    if (header.token_count == 0) {
        fprintf(stderr, "Error: dataset contains zero tokens\n");
        goto fail;
    }

    /* Overflow-safe allocation size check — token_count is a 64-bit
     * value straight from the file; on a 32-bit size_t platform in
     * particular this must be checked before multiplying, not after. */
    if ((uint64_t)header.token_count > (uint64_t)(SIZE_MAX / sizeof(uint32_t))) {
        fprintf(stderr, "Error: token_count too large for this platform\n");
        goto fail;
    }
    size_t token_count = (size_t)header.token_count;

    /* ── token array ─────────────────────────────────────────────── */
    ds->tokens = (uint32_t *)malloc(token_count * sizeof(uint32_t));
    /* token_count * sizeof(uint32_t) is safe here because token_count
     * was just bounds-checked against SIZE_MAX / sizeof(uint32_t)
     * above. */
    if (!ds->tokens) {
        fprintf(stderr,
            "Error: out of memory allocating %zu tokens\n", token_count);
        goto fail;
    }

    if (!read_exact(f, ds->tokens, token_count, sizeof(uint32_t), "token array"))
        goto fail;

    /* Strict end-of-file check: the format has no length beyond the
     * header's token_count, so any bytes left after the expected
     * token array mean a corrupted file or a writer/reader version
     * mismatch — treat it as an error rather than silently ignoring
     * the extra data. */
    {
        int c = fgetc(f);
        if (c != EOF) {
            fprintf(stderr, "Error: extra unexpected data at end of file\n");
            goto fail;
        }
    }

    ds->token_count = token_count;
    ds->pos         = 0;

    fclose(f);
    ds->f = NULL;   /* fully loaded into memory — no file I/O needed after this */
    return ds;

fail:
    dataset_close(ds);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════
 *  dataset_next_batch
 * ═══════════════════════════════════════════════════════════════════
 * Pure in-memory operation: batch_size sequential memcpy pairs, no
 * allocation, no file I/O, no per-token modulo. The wrap check is a
 * single comparison per batch element (not per token), so throughput
 * stays sequential-memory-bound.
 */

int dataset_next_batch(Dataset *ds, uint32_t *input, uint32_t *target,
                       size_t batch_size, size_t seq_len) {
    if (!ds || !input || !target) {
        fprintf(stderr, "Error: null argument passed to dataset_next_batch\n");
        return -1;
    }
    if (batch_size == 0 || seq_len == 0) {
        fprintf(stderr, "Error: batch_size and seq_len must both be > 0\n");
        return -1;
    }
    /* Overflow-safe check on the *output layout* stride (b * seq_len);
     * this function itself allocates nothing, but an overflowing
     * stride would still produce out-of-bounds writes into the
     * caller's buffers. */
    if (seq_len != 0 && batch_size > SIZE_MAX / seq_len) {
        fprintf(stderr, "Error: batch_size * seq_len overflows size_t\n");
        return -1;
    }

    /* A (seq_len -> seq_len) pair needs seq_len+1 contiguous tokens
     * (input covers [pos, pos+seq_len-1], target covers
     * [pos+1, pos+seq_len]). If the whole dataset can't supply that
     * even once, no valid batch can ever be produced — fail instead
     * of wrapping into an infinite/out-of-bounds loop. */
    if (ds->token_count < seq_len + 1) {
        fprintf(stderr,
            "Error: dataset has %zu tokens, too few for seq_len %zu "
            "(need at least %zu)\n",
            ds->token_count, seq_len, seq_len + 1);
        return -1;
    }

    for (size_t b = 0; b < batch_size; b++) {
        if (ds->pos + seq_len >= ds->token_count) {
            ds->pos = 0;
        }

        uint32_t *input_ptr  = input  + b * seq_len;
        uint32_t *target_ptr = target + b * seq_len;

        memcpy(input_ptr,  &ds->tokens[ds->pos],     seq_len * sizeof(uint32_t));
        memcpy(target_ptr, &ds->tokens[ds->pos + 1], seq_len * sizeof(uint32_t));

        ds->pos += seq_len;
    }

    return 0;
}
