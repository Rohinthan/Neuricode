#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── big-endian helper ───────────────────────────────────────────── */

static unsigned int read_be32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((unsigned int)b[0] << 24) |
           ((unsigned int)b[1] << 16) |
           ((unsigned int)b[2] <<  8) |
            (unsigned int)b[3];
}

/* ── mnist_load ──────────────────────────────────────────────────── */

MNISTData *mnist_load(const char *img_path, const char *lbl_path,
                       int max_samples) {
    /* open files */
    FILE *fi = fopen(img_path, "rb");
    if (!fi) {
        fprintf(stderr, "[mnist] Cannot open images: %s\n", img_path);
        fprintf(stderr, "[mnist] Download with: bash mnist/download.sh\n");
        return NULL;
    }
    FILE *fl = fopen(lbl_path, "rb");
    if (!fl) {
        fprintf(stderr, "[mnist] Cannot open labels: %s\n", lbl_path);
        fclose(fi);
        return NULL;
    }

    /* read image header */
    unsigned int magic_img = read_be32(fi);
    if (magic_img != 2051) {
        fprintf(stderr, "[mnist] Bad image magic: %u (expected 2051)\n",
                magic_img);
        fclose(fi); fclose(fl); return NULL;
    }
    unsigned int n_img  = read_be32(fi);
    unsigned int rows   = read_be32(fi);
    unsigned int cols   = read_be32(fi);
    unsigned int pixels = rows * cols;  /* 784 for standard MNIST */

    /* read label header */
    unsigned int magic_lbl = read_be32(fl);
    if (magic_lbl != 2049) {
        fprintf(stderr, "[mnist] Bad label magic: %u (expected 2049)\n",
                magic_lbl);
        fclose(fi); fclose(fl); return NULL;
    }
    unsigned int n_lbl = read_be32(fl);

    if (n_img != n_lbl) {
        fprintf(stderr, "[mnist] Image/label count mismatch: %u vs %u\n",
                n_img, n_lbl);
        fclose(fi); fclose(fl); return NULL;
    }

    /* cap at max_samples */
    int n = (int)n_img;
    if (max_samples > 0 && max_samples < n) n = max_samples;

    /* allocate MNISTData */
    MNISTData *d = (MNISTData *)malloc(sizeof(MNISTData));
    CF_CHECK_ALLOC(d);
    d->n = n;

    int sh_img[2] = {n, (int)pixels};
    int sh_lbl[2] = {n, 10};
    int sh_raw[2] = {n, 1};

    d->images     = tensor_zeros(sh_img, 2);
    d->labels     = tensor_zeros(sh_lbl, 2);
    d->labels_raw = tensor_zeros(sh_raw, 2);

    /* read pixel data */
    unsigned char *buf = (unsigned char *)malloc(pixels);
    CF_CHECK_ALLOC(buf);

    for (int i = 0; i < n; i++) {
        if (fread(buf, 1, pixels, fi) != pixels) {
            fprintf(stderr, "[mnist] Unexpected EOF at image %d\n", i);
            break;
        }
        /* normalize [0,255] → [0,1] */
        for (unsigned int p = 0; p < pixels; p++)
            d->images->data[i * pixels + p] = buf[p] / 255.0f;
    }
    free(buf);

    /* read label data */
    for (int i = 0; i < n; i++) {
        unsigned char lbl;
        if (fread(&lbl, 1, 1, fl) != 1) break;
        d->labels_raw->data[i] = (float)lbl;
        /* one-hot encode */
        d->labels->data[i * 10 + lbl] = 1.0f;
    }

    fclose(fi);
    fclose(fl);
    return d;
}

void mnist_free(MNISTData *d) {
    if (!d) return;
    tensor_free(d->images);
    tensor_free(d->labels);
    tensor_free(d->labels_raw);
    free(d);
}

/* ── mnist_print_sample ──────────────────────────────────────────── */

void mnist_print_sample(const MNISTData *d, int idx) {
    if (idx < 0 || idx >= d->n) return;
    int label = (int)d->labels_raw->data[idx];
    printf("Sample %d — Label: %d\n", idx, label);
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            float v = d->images->data[idx * 784 + r * 28 + c];
            /* ASCII art: dark pixels = filled */
            if      (v > 0.75f) printf("██");
            else if (v > 0.50f) printf("▓▓");
            else if (v > 0.25f) printf("░░");
            else                printf("  ");
        }
        printf("\n");
    }
}

/* ── mnist_get_batch ─────────────────────────────────────────────── */

void mnist_get_batch(const MNISTData *d, int start,
                     int batch_size, Tensor *X, Tensor *Y) {
    int actual = batch_size;
    if (start + actual > d->n) actual = d->n - start;

    memcpy(X->data,
           d->images->data + start * 784,
           actual * 784 * sizeof(float));
    memcpy(Y->data,
           d->labels->data + start * 10,
           actual * 10 * sizeof(float));
}

/* ── mnist_shuffle ───────────────────────────────────────────────── */

void mnist_shuffle(MNISTData *d) {
    int n = d->n;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        /* swap image rows */
        float tmp_img[784];
        memcpy(tmp_img,
               d->images->data + i * 784, 784 * sizeof(float));
        memcpy(d->images->data + i * 784,
               d->images->data + j * 784, 784 * sizeof(float));
        memcpy(d->images->data + j * 784,
               tmp_img,                   784 * sizeof(float));

        /* swap label rows */
        float tmp_lbl[10];
        memcpy(tmp_lbl,
               d->labels->data + i * 10, 10 * sizeof(float));
        memcpy(d->labels->data + i * 10,
               d->labels->data + j * 10, 10 * sizeof(float));
        memcpy(d->labels->data + j * 10,
               tmp_lbl,                  10 * sizeof(float));

        /* swap raw label */
        float tmp_raw = d->labels_raw->data[i];
        d->labels_raw->data[i] = d->labels_raw->data[j];
        d->labels_raw->data[j] = tmp_raw;
    }
}

/* ── mnist_print_info ────────────────────────────────────────────── */

void mnist_print_info(const MNISTData *d, const char *name) {
    printf("MNIST '%s': %d samples, 784 features, 10 classes\n",
           name ? name : "?", d->n);

    /* count per class */
    int counts[10] = {0};
    for (int i = 0; i < d->n; i++)
        counts[(int)d->labels_raw->data[i]]++;
    printf("  Class distribution: ");
    for (int c = 0; c < 10; c++)
        printf("%d:%d ", c, counts[c]);
    printf("\n");
}
