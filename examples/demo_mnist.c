/*
 * demo_mnist.c — neuralc MNIST handwritten digit recognition
 *
 * Network: 784 -> 256 ReLU -> 128 ReLU -> 10 Softmax
 * Optimizer: Adam
 *
 * Uses the polymorphic Network API throughout:
 *   nn_set_input_shape, nn_add_dense, nn_forward,
 *   nn_backward, nn_train_step, adam_step
 *
 * No direct layer access required.
 *
 * Build:  make mnist_demo
 * Run:    ./mnist_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensor.h"
#include "layer.h"
#include "nn.h"
#include "optimizer.h"
#include "mnist.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

#ifdef BATCH_SIZE
#undef BATCH_SIZE
#endif
#ifdef NEURALC_BATCH_SIZE
  #define BATCH_SIZE NEURALC_BATCH_SIZE
#else
  #define BATCH_SIZE 64
#endif

#ifdef EPOCHS
#undef EPOCHS
#endif
#ifdef NEURALC_EPOCHS
  #define EPOCHS NEURALC_EPOCHS
#else
  #define EPOCHS 20
#endif

#ifdef LR
#undef LR
#endif
#ifdef NEURALC_LR
  #define LR NEURALC_LR
#else
  #define LR 0.001f
#endif

/* ── evaluate accuracy on a dataset ─────────────────────────────── */
static float evaluate(Network *net, MNISTData *data) {
    int correct = 0;

    for (int s = 0; s < data->n; s += BATCH_SIZE) {
        int actual = BATCH_SIZE;
        if (s + actual > data->n) actual = data->n - s;

        int shX[2] = {actual, 784};
        int shY[2] = {actual, 10};
        Tensor *X   = tensor_zeros(shX, 2);
        Tensor *Y   = tensor_zeros(shY, 2);
        Tensor *out = tensor_zeros(shY, 2);

        memcpy(X->data,
               data->images->data + s * 784,
               (size_t)actual * 784 * sizeof(float));
        memcpy(Y->data,
               data->labels->data + s * 10,
               (size_t)actual * 10 * sizeof(float));

        nn_forward(net, X, out, /*training=*/0);

        int preds[BATCH_SIZE], trues[BATCH_SIZE];
        tensor_argmax_rows(out, preds);
        tensor_argmax_rows(Y,   trues);
        for (int i = 0; i < actual; i++)
            if (preds[i] == trues[i]) correct++;

        tensor_free(X);
        tensor_free(Y);
        tensor_free(out);
    }
    return (float)correct / data->n * 100.0f;
}

/* ── main ────────────────────────────────────────────────────────── */
int main(void) {
    srand((unsigned)time(NULL));

    printf("=== neuralc MNIST Digit Recognition ===\n");
#ifdef USE_OMP
    printf("  [OpenMP] Active Threads: %d\n", omp_get_max_threads());
#else
    printf("  [OpenMP] Disabled (Single Threaded)\n");
#endif
#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
    printf("  [Config] Batch Size: %d, Epochs: %d, LR: %.6f\n", BATCH_SIZE, EPOCHS, LR);
#endif
    printf("\n");

    /* load data */
    printf("Loading MNIST...\n");
    MNISTData *train = mnist_load(
        "mnist/train-images-idx3-ubyte",
        "mnist/train-labels-idx1-ubyte", 0);
    MNISTData *test = mnist_load(
        "mnist/t10k-images-idx3-ubyte",
        "mnist/t10k-labels-idx1-ubyte", 0);

    if (!train || !test) {
        fprintf(stderr, "Run: bash mnist/download.sh\n");
        return 1;
    }

    printf("  Loaded %d train, %d test\n", train->n, test->n);
    mnist_print_info(train, "train");
    mnist_print_info(test,  "test");

    printf("\nSample digit:\n");
    mnist_print_sample(train, 0);

    /* build network */
    printf("\nBuilding network...\n");
    Network *net = nn_create();

    DenseLayer *l1 = dense_create(784, 256, ACT_RELU);
    DenseLayer *l2 = dense_create(256, 128, ACT_RELU);
    DenseLayer *l3 = dense_create(128, 10, ACT_SOFTMAX);

    nn_add_dense(net, l1);
    nn_add_dense(net, l2);
    nn_add_dense(net, l3);

    Adam *opt = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1e-4f);

    int n_batches = (train->n + BATCH_SIZE - 1) / BATCH_SIZE;

    printf("\n  %-6s %-12s %-12s %-10s\n",
           "Epoch", "Train Loss", "Train Acc", "Test Acc");
    printf("  %s\n",
           "────────────────────────────────────────────");

    float best_acc = 0.0f;

    for (int ep = 1; ep <= EPOCHS; ep++) {
        mnist_shuffle(train);
        float total_loss = 0.0f;
        int   correct    = 0;

        for (int b = 0; b < n_batches; b++) {
            int start  = b * BATCH_SIZE;
            int actual = BATCH_SIZE;
            if (start + actual > train->n) actual = train->n - start;

            int shX[2] = {actual, 784};
            int shY[2] = {actual, 10};
            Tensor *X    = tensor_zeros(shX, 2);
            Tensor *Y    = tensor_zeros(shY, 2);
            Tensor *pred = tensor_zeros(shY, 2);
            Tensor *grad = tensor_zeros(shY, 2);

            memcpy(X->data,
                   train->images->data + start * 784,
                   (size_t)actual * 784 * sizeof(float));
            memcpy(Y->data,
                   train->labels->data + start * 10,
                   (size_t)actual * 10 * sizeof(float));

            float loss = nn_train_step(net, X, Y,
                                       LOSS_CROSS_ENTROPY,
                                       pred, grad);
            total_loss += loss;
            adam_step(opt, net);

            int preds[BATCH_SIZE], trues[BATCH_SIZE];
            tensor_argmax_rows(pred, preds);
            tensor_argmax_rows(Y,    trues);
            for (int i = 0; i < actual; i++)
                if (preds[i] == trues[i]) correct++;

            tensor_free(X); tensor_free(Y);
            tensor_free(pred); tensor_free(grad);
        }

        float train_loss = total_loss / n_batches;
        float train_acc  = (float)correct / train->n * 100.0f;
        float test_acc   = evaluate(net, test);

        if (test_acc > best_acc) {
            best_acc = test_acc;
            nn_save(net, "mnist_best.bin");
        }

        printf("  %-6d %-12.4f %-12.1f%% %-10.1f%%\n",
               ep, train_loss, train_acc, test_acc);
    }

    printf("\n  Best test accuracy: %.2f%%\n", best_acc);
    printf("  Model saved: mnist_best.bin\n");

    /* sample predictions */
    printf("\n  Sample Predictions:\n");
    printf("  %-6s %-10s %-10s\n", "Index", "Predicted", "Actual");

    for (int i = 0; i < 10; i++) {
        int shx[2] = {1, 784};
        int shy[2] = {1, 10};
        Tensor *xi = tensor_zeros(shx, 2);
        Tensor *pi = tensor_zeros(shy, 2);
        memcpy(xi->data,
               test->images->data + i * 784,
               784 * sizeof(float));
        nn_forward(net, xi, pi, /*training=*/0);
        int pred   = tensor_argmax(pi);
        int actual = (int)test->labels_raw->data[i];
        printf("  %-6d %-10d %-10d %s\n",
               i, pred, actual, pred == actual ? "+" : "x");
        tensor_free(xi);
        tensor_free(pi);
    }

    printf("\ndone\n\n");

    adam_free(opt);
    nn_free(net);
    mnist_free(train);
    mnist_free(test);
    return 0;
}
