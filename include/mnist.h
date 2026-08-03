#ifndef MNIST_H
#define MNIST_H

#include "tensor.h"

/*
 * mnist.h — MNIST dataset loader for neuralc
 *
 * Loads the original MNIST binary IDX format files.
 *
 * Download dataset (4 files) into a mnist/ folder:
 *   wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
 *   wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
 *   wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
 *   wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
 *   cd mnist && gunzip *.gz
 *
 * File format:
 *   Images: [magic][n_images][rows][cols][pixels...]
 *   Labels: [magic][n_labels][labels...]
 *   All integers are big-endian 32-bit.
 *   Pixels are uint8 [0,255] — we normalize to [0,1].
 */

typedef struct {
    Tensor *images;     /* [n, 784]  float32 normalized [0,1] */
    Tensor *labels;     /* [n, 10]   one-hot float32           */
    Tensor *labels_raw; /* [n, 1]    integer class 0-9         */
    int     n;          /* number of samples                   */
} MNISTData;

/* ── load ───────────────────────────────────────────────────────── */

/*
 * mnist_load:
 *   img_path  — path to images file e.g. "mnist/train-images-idx3-ubyte"
 *   lbl_path  — path to labels file e.g. "mnist/train-labels-idx1-ubyte"
 *   max_samples — max to load (0 = load all)
 *
 * Returns NULL on failure.
 */
MNISTData *mnist_load(const char *img_path, const char *lbl_path,
                       int max_samples);

void mnist_free(MNISTData *d);

/* ── utilities ──────────────────────────────────────────────────── */

/* Print a sample as ASCII art (28x28) */
void mnist_print_sample(const MNISTData *d, int idx);

/* Get a mini-batch: fills X [batch,784] and Y [batch,10] */
void mnist_get_batch(const MNISTData *d, int start,
                     int batch_size, Tensor *X, Tensor *Y);

/* Shuffle the dataset (Fisher-Yates) */
void mnist_shuffle(MNISTData *d);

/* Print dataset info */
void mnist_print_info(const MNISTData *d, const char *name);

#endif /* MNIST_H */
