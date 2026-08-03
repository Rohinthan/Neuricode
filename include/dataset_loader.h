#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

/*
 * dataset_loader.h — sequential loader for the fixed binary token
 * dataset format (dataset.bin). See dataset_loader.c for the exact
 * on-disk layout and validation rules.
 *
 * Usage pattern:
 *   Dataset *ds = dataset_open("dataset.bin");
 *   if (!ds) { ... handle error ... }
 *
 *   uint32_t *input  = malloc(batch_size * seq_len * sizeof(uint32_t));
 *   uint32_t *target = malloc(batch_size * seq_len * sizeof(uint32_t));
 *   // allocate these ONCE, outside any training loop
 *
 *   while (training) {
 *       if (dataset_next_batch(ds, input, target, batch_size, seq_len) != 0)
 *           break;   // dataset too small for even one (batch,seq) pair
 *       // ... use input/target ...
 *   }
 *
 *   free(input); free(target);
 *   dataset_close(ds);
 */

#define DATASET_MAGIC 0x44535431u  /* "DST1" */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;        /* must be DATASET_MAGIC */
    uint32_t version;      /* must be 1             */
    uint64_t token_count;  /* total number of tokens */
    uint32_t reserved[5];  /* must all be 0          */
} DatasetHeader;
#pragma pack(pop)

typedef struct {
    FILE *f;                /* kept open only for the lifetime of the
                              * struct; the whole dataset is read into
                              * `tokens` up front, so no further I/O
                              * happens through this handle after
                              * dataset_open() returns. */

    uint32_t *tokens;       /* full dataset, in memory, contiguous   */
    size_t    token_count;

    size_t    pos;          /* current streaming position            */
} Dataset;

/*
 * dataset_open: opens `path`, reads and validates the DatasetHeader,
 * then reads the entire token array into memory in one pass.
 *
 * Returns a fully populated Dataset* on success (pos == 0).
 * Returns NULL on any failure (bad magic/version/reserved fields,
 * empty dataset, short read, trailing garbage, or allocation
 * failure) — a clear message is printed to stderr and everything
 * allocated so far is freed before returning.
 */
Dataset *dataset_open(const char *path);

/*
 * dataset_close: releases the file handle and the token buffer, then
 * the Dataset struct itself. Safe to call with ds == NULL (no-op).
 */
void dataset_close(Dataset *ds);

/*
 * dataset_next_batch: fills `input`/`target` with `batch_size`
 * contiguous (input, target) sequence pairs of length `seq_len`,
 * advancing and wrapping ds->pos as needed.
 *
 * For each of the batch_size slots:
 *   input [b*seq_len .. b*seq_len+seq_len-1] = tokens[pos   .. pos+seq_len-1]
 *   target[b*seq_len .. b*seq_len+seq_len-1] = tokens[pos+1 .. pos+seq_len]
 *   pos += seq_len   (wrapping to 0 first if pos+seq_len would reach
 *                     or exceed token_count, so every copy above stays
 *                     in bounds)
 *
 * `input` and `target` must each point to caller-allocated buffers of
 * at least batch_size * seq_len uint32_t elements; this function
 * performs no allocation of its own and does no file I/O (the dataset
 * is already fully resident in memory from dataset_open()).
 *
 * Returns 0 on success. Returns -1 if the dataset does not contain
 * enough tokens to produce even one (seq_len -> seq_len) pair
 * (token_count < seq_len + 1), or if arguments are invalid (NULL
 * pointers, batch_size == 0, seq_len == 0, or batch_size * seq_len
 * would overflow size_t) — no partial batch is ever written in that
 * case.
 */
int dataset_next_batch(Dataset *ds, uint32_t *input, uint32_t *target,
                       size_t batch_size, size_t seq_len);

#endif /* DATASET_LOADER_H */
