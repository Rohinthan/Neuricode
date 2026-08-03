#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"

/*
 * Dataset: holds features X and labels Y loaded from a CSV file.
 *
 * CSV format expected:
 *   - First row: optional header (auto-detected)
 *   - Each subsequent row: feature1, feature2, ..., featureN, label
 *   - Last column is always treated as the label
 */
typedef struct {
    Tensor *X;          /* [num_samples, num_features] */
    Tensor *Y;          /* [num_samples, 1]            */
    int     num_samples;
    int     num_features;
} Dataset;

/* ── load / free ────────────────────────────────────────────────── */

/* Load CSV from file path. has_header=1 skips first row.           */
Dataset *dataset_load_csv(const char *path, int has_header);
void     dataset_free(Dataset *ds);

/* ── one-hot encoding ───────────────────────────────────────────── */
/* Convert integer label column [N,1] → one-hot [N, num_classes]   */
Tensor  *dataset_one_hot(const Tensor *labels, int num_classes);

/* ── train/val split ────────────────────────────────────────────── */
/* Splits dataset into train and val by ratio (e.g. 0.8 = 80% train) */
void dataset_split(const Dataset *ds, float train_ratio,
                   Dataset *train_out, Dataset *val_out);

/* ── shuffle ────────────────────────────────────────────────────── */
void dataset_shuffle(Dataset *ds);

/* ── normalize ──────────────────────────────────────────────────── */
/* Min-max normalize X to [0, 1] per feature column                 */
void dataset_normalize_minmax(Dataset *ds);
/* Z-score normalize X (zero mean, unit variance) per feature       */
void dataset_normalize_zscore(Dataset *ds);

/* ── batch iterator ─────────────────────────────────────────────── */
typedef struct {
    const Dataset *ds;
    int            batch_size;
    int            current;     /* current sample index             */
} BatchIter;

BatchIter batch_iter_create(const Dataset *ds, int batch_size);
/* Returns 1 if a batch was filled, 0 if epoch is done             */
int  batch_iter_next(BatchIter *it, Tensor *X_batch, Tensor *Y_batch);
void batch_iter_reset(BatchIter *it);

/* ── print info ─────────────────────────────────────────────────── */
void dataset_print_info(const Dataset *ds, const char *name);

#endif /* DATALOADER_H */
