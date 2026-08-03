#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LEN  4096
#define MAX_COLS      512

/* ── helpers ─────────────────────────────────────────────────────── */

/* Count commas in a line → number of columns */
static int count_cols(const char *line) {
    int c = 1;
    while (*line) { if (*line++ == ',') c++; }
    return c;
}

/* Count data rows (excluding header if present) */
static int count_rows(FILE *f, int has_header) {
    char line[MAX_LINE_LEN];
    int rows = 0;
    rewind(f);
    while (fgets(line, sizeof(line), f)) rows++;
    return has_header ? rows - 1 : rows;
}

static int is_header(const char *line) {
    /* heuristic: if first token is not a number, it's a header */
    char tmp[64];
    sscanf(line, "%63[^,\n]", tmp);
    char *end;
    strtod(tmp, &end);
    return (end == tmp); /* strtod didn't consume anything → not a number */
}

/* ── dataset_load_csv ────────────────────────────────────────────── */

Dataset *dataset_load_csv(const char *path, int has_header) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[cforge] dataset_load_csv: cannot open '%s'\n", path);
        return NULL;
    }

    char line[MAX_LINE_LEN];

    /* auto-detect header */
    if (fgets(line, sizeof(line), f)) {
        if (!has_header && is_header(line))
            has_header = 1;   /* auto-skip */
        else if (!has_header)
            rewind(f);        /* first row is data, rewind */
    }

    int num_cols = 0;
    /* peek at first data line to get column count */
    long data_start = ftell(f);
    if (fgets(line, sizeof(line), f))
        num_cols = count_cols(line);
    fseek(f, data_start, SEEK_SET);

    int num_features = num_cols - 1;  /* last col = label */
    int num_samples  = count_rows(f, 0); /* count from current position */
    fseek(f, data_start, SEEK_SET);

    /* allocate tensors */
    int shX[2] = {num_samples, num_features};
    int shY[2] = {num_samples, 1};
    Tensor *X = tensor_zeros(shX, 2);
    Tensor *Y = tensor_zeros(shY, 2);
    if (!X || !Y) {
        tensor_free(X); tensor_free(Y);
        fclose(f); return NULL;
    }

    /* parse rows */
    int row = 0;
    while (row < num_samples && fgets(line, sizeof(line), f)) {
        char *tok = strtok(line, ",\n\r");
        int   col = 0;
        while (tok && col < num_cols) {
            float val = (float)atof(tok);
            if (col < num_features)
                X->data[row * num_features + col] = val;
            else
                Y->data[row] = val;
            col++;
            tok = strtok(NULL, ",\n\r");
        }
        row++;
    }

    fclose(f);

    Dataset *ds = (Dataset *)malloc(sizeof(Dataset));
    ds->X            = X;
    ds->Y            = Y;
    ds->num_samples  = num_samples;
    ds->num_features = num_features;
    return ds;
}

void dataset_free(Dataset *ds) {
    if (!ds) return;
    tensor_free(ds->X);
    tensor_free(ds->Y);
    free(ds);
}

/* ── one-hot encoding ────────────────────────────────────────────── */

Tensor *dataset_one_hot(const Tensor *labels, int num_classes) {
    int n   = (int)labels->size;
    int sh[2] = {n, num_classes};
    Tensor *out = tensor_zeros(sh, 2);
    for (int i = 0; i < n; i++) {
        int cls = (int)labels->data[i];
        if (cls >= 0 && cls < num_classes)
            out->data[i * num_classes + cls] = 1.0f;
    }
    return out;
}

/* ── train/val split ─────────────────────────────────────────────── */

void dataset_split(const Dataset *ds, float train_ratio,
                   Dataset *train_out, Dataset *val_out) {
    int n_train = (int)(ds->num_samples * train_ratio);
    int n_val   = ds->num_samples - n_train;
    int nf      = ds->num_features;

    int shXtr[2] = {n_train, nf}, shYtr[2] = {n_train, 1};
    int shXva[2] = {n_val,   nf}, shYva[2] = {n_val,   1};

    train_out->X = tensor_zeros(shXtr, 2);
    train_out->Y = tensor_zeros(shYtr, 2);
    val_out->X   = tensor_zeros(shXva, 2);
    val_out->Y   = tensor_zeros(shYva, 2);
    train_out->num_samples  = n_train;
    train_out->num_features = nf;
    val_out->num_samples    = n_val;
    val_out->num_features   = nf;

    /* copy train rows */
    memcpy(train_out->X->data, ds->X->data, n_train * nf * sizeof(float));
    memcpy(train_out->Y->data, ds->Y->data, n_train * sizeof(float));
    /* copy val rows */
    memcpy(val_out->X->data, ds->X->data + n_train * nf,
           n_val * nf * sizeof(float));
    memcpy(val_out->Y->data, ds->Y->data + n_train,
           n_val * sizeof(float));
}

/* ── shuffle ─────────────────────────────────────────────────────── */

void dataset_shuffle(Dataset *ds) {
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    int nf = ds->num_features;
    int n  = ds->num_samples;
    /* Fisher-Yates */
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        /* swap row i and row j in X */
        for (int c = 0; c < nf; c++) {
            float tmp = ds->X->data[i*nf+c];
            ds->X->data[i*nf+c] = ds->X->data[j*nf+c];
            ds->X->data[j*nf+c] = tmp;
        }
        /* swap label */
        float tmp = ds->Y->data[i];
        ds->Y->data[i] = ds->Y->data[j];
        ds->Y->data[j] = tmp;
    }
}

/* ── normalize ───────────────────────────────────────────────────── */

void dataset_normalize_minmax(Dataset *ds) {
    int n = ds->num_samples, nf = ds->num_features;
    for (int c = 0; c < nf; c++) {
        float mn = ds->X->data[c], mx = ds->X->data[c];
        for (int r = 1; r < n; r++) {
            float v = ds->X->data[r*nf+c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float range = mx - mn;
        if (range < 1e-8f) range = 1.0f;
        for (int r = 0; r < n; r++)
            ds->X->data[r*nf+c] = (ds->X->data[r*nf+c] - mn) / range;
    }
}

void dataset_normalize_zscore(Dataset *ds) {
    int n = ds->num_samples, nf = ds->num_features;
    for (int c = 0; c < nf; c++) {
        /* mean */
        float mu = 0.0f;
        for (int r = 0; r < n; r++) mu += ds->X->data[r*nf+c];
        mu /= n;
        /* std */
        float var = 0.0f;
        for (int r = 0; r < n; r++) {
            float d = ds->X->data[r*nf+c] - mu;
            var += d * d;
        }
        float std = sqrtf(var / n);
        if (std < 1e-8f) std = 1.0f;
        for (int r = 0; r < n; r++)
            ds->X->data[r*nf+c] = (ds->X->data[r*nf+c] - mu) / std;
    }
}

/* ── batch iterator ──────────────────────────────────────────────── */

BatchIter batch_iter_create(const Dataset *ds, int batch_size) {
    BatchIter it;
    it.ds         = ds;
    it.batch_size = batch_size;
    it.current    = 0;
    return it;
}

int batch_iter_next(BatchIter *it, Tensor *X_batch, Tensor *Y_batch) {
    int n  = it->ds->num_samples;
    int nf = it->ds->num_features;
    if (it->current >= n) return 0;

    int actual = it->batch_size;
    if (it->current + actual > n) actual = n - it->current;

    memcpy(X_batch->data,
           it->ds->X->data + it->current * nf,
           actual * nf * sizeof(float));
    memcpy(Y_batch->data,
           it->ds->Y->data + it->current,
           actual * sizeof(float));

    it->current += actual;
    return 1;
}

void batch_iter_reset(BatchIter *it) { it->current = 0; }

/* ── print info ──────────────────────────────────────────────────── */

void dataset_print_info(const Dataset *ds, const char *name) {
    printf("Dataset '%s': %d samples, %d features\n",
           name ? name : "?", ds->num_samples, ds->num_features);
    printf("  X shape: [%d, %d]  Y shape: [%d, 1]\n",
           ds->num_samples, ds->num_features, ds->num_samples);
}
