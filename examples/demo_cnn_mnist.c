/*
 * demo_cnn_mnist.c — CNN pipeline for MNIST digit recognition
 *
 * Architecture (from analysis manifest):
 *   [B, 1, 28, 28]
 *   Conv2D(1->8, 3x3, pad=0)  -> [B, 8, 26, 26]
 *   ReLU
 *   MaxPool2D(2x2, stride=2)  -> [B, 8, 13, 13]
 *   Conv2D(8->16, 3x3, pad=0) -> [B, 16, 11, 11]
 *   ReLU
 *   MaxPool2D(2x2, stride=2)  -> [B, 16, 5, 5]
 *   Flatten                   -> [B, 400]
 *   Dense(400->128)
 *   ReLU
 *   Dense(128->10, Softmax)
 *
 * All intermediate tensors are pre-allocated at build time.
 * No malloc occurs inside the training loop.
 *
 * Build:  make cnn_mnist
 * Run:    ./cnn_mnist
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

/*
 * Pull defaults from menuconfig output when available.
 * Falls back to built-in values when neuralc_config.h
 * has not been generated yet (make config).
 */
#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

#ifndef BATCH_SIZE
#ifdef NEURALC_BATCH_SIZE
  #define BATCH_SIZE  NEURALC_BATCH_SIZE
#else
  #define BATCH_SIZE  32
#endif
#endif

#ifndef EPOCHS
#ifdef NEURALC_EPOCHS
  #define EPOCHS      NEURALC_EPOCHS
#else
  #define EPOCHS      10
#endif
#endif

#ifndef LR
#ifdef NEURALC_LR
  #define LR          NEURALC_LR
#else
  #define LR          0.001f
#endif
#endif

static float evaluate(Network *net, MNISTData *data) {
    int correct = 0;
    int sh_out[2] = {BATCH_SIZE, 10};
    Tensor *out = tensor_zeros(sh_out, 2);

    for (int s = 0; s < data->n; s += BATCH_SIZE) {
        int actual = BATCH_SIZE;
        if (s + actual > data->n) actual = data->n - s;

        /* build 4D input [actual, 1, 28, 28] */
        int shX[4] = {actual, 1, 28, 28};
        int shY[2] = {actual, 10};
        Tensor *X = tensor_zeros(shX, 4);
        Tensor *Y = tensor_zeros(shY, 2);

        /* copy flat 784 pixels into 4D layout */
        memcpy(X->data,
               data->images->data + s * 784,
               (size_t)actual * 784 * sizeof(float));
        memcpy(Y->data,
               data->labels->data + s * 10,
               (size_t)actual * 10 * sizeof(float));

        /* resize output if last batch is smaller */
        if (actual != BATCH_SIZE) {
            tensor_free(out);
            int sh2[2] = {actual, 10};
            out = tensor_zeros(sh2, 2);
        }

        nn_forward(net, X, out);

        int preds[BATCH_SIZE], trues[BATCH_SIZE];
        tensor_argmax_rows(out, preds);
        tensor_argmax_rows(Y,   trues);
        for (int i = 0; i < actual; i++)
            if (preds[i] == trues[i]) correct++;

        tensor_free(X);
        tensor_free(Y);
    }

    tensor_free(out);
    return (float)correct / data->n * 100.0f;
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== neuralc CNN MNIST Demo ===\n\n");

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
    mnist_print_info(train, "train");
    mnist_print_info(test,  "test");

    /*
     * Build CNN pipeline.
     * Each layer_xxx() call resolves and stores the output shape
     * so the network can pre-allocate workspace buffers.
     */
    Network *net = nn_create();

    /* input shape: [batch, 1, 28, 28] */
    int in4[4] = {BATCH_SIZE, 1, 28, 28};
    nn_set_input_shape(net, in4, 4);

    /* layer 0: Conv2D 1->8, 3x3, pad=0 => [B, 8, 26, 26] */
    nn_add_layer(net, layer_conv2d(1, 8, 3, 1, 0, in4, 4));

    /* layer 1: ReLU */
    nn_add_layer(net, layer_relu(net->layers[0].out_shape,
                                  net->layers[0].out_ndim));

    /* layer 2: MaxPool 2x2 stride 2 => [B, 8, 13, 13] */
    nn_add_layer(net, layer_maxpool2d(2, 2,
                                      net->layers[1].out_shape,
                                      net->layers[1].out_ndim));

    /* layer 3: Conv2D 8->16, 3x3, pad=0 => [B, 16, 11, 11] */
    nn_add_layer(net, layer_conv2d(8, 16, 3, 1, 0,
                                   net->layers[2].out_shape,
                                   net->layers[2].out_ndim));

    /* layer 4: ReLU */
    nn_add_layer(net, layer_relu(net->layers[3].out_shape,
                                  net->layers[3].out_ndim));

    /* layer 5: MaxPool 2x2 stride 2 => [B, 16, 5, 5] */
    nn_add_layer(net, layer_maxpool2d(2, 2,
                                      net->layers[4].out_shape,
                                      net->layers[4].out_ndim));

    /* layer 6: Flatten => [B, 400] */
    nn_add_layer(net, layer_flatten(net->layers[5].out_shape,
                                     net->layers[5].out_ndim));

    /* layer 7: Dense 400->128 */
    nn_add_layer(net, layer_dense(400, 128, ACT_NONE,
                                   net->layers[6].out_shape,
                                   net->layers[6].out_ndim));

    /* layer 8: ReLU */
    nn_add_layer(net, layer_relu(net->layers[7].out_shape,
                                  net->layers[7].out_ndim));

    /* layer 9: Dense 128->10, Softmax */
    nn_add_layer(net, layer_dense(128, 10, ACT_SOFTMAX,
                                   net->layers[8].out_shape,
                                   net->layers[8].out_ndim));

    nn_print_summary(net);

    /* optimizer */
    Adam *opt = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1e-4f);

    int n_batches = (train->n + BATCH_SIZE - 1) / BATCH_SIZE;
    int sh_out[2] = {BATCH_SIZE, 10};
    Tensor *pred = tensor_zeros(sh_out, 2);
    Tensor *grad = tensor_zeros(sh_out, 2);

    printf("\n  %-6s %-12s %-12s %-10s\n",
           "Epoch", "Train Loss", "Train Acc", "Test Acc");
    printf("  %s\n", "──────────────────────────────────────────");

    for (int ep = 1; ep <= EPOCHS; ep++) {
        mnist_shuffle(train);
        float total_loss = 0.0f;
        int   correct    = 0;

        for (int b = 0; b < n_batches; b++) {
            int start  = b * BATCH_SIZE;
            int actual = BATCH_SIZE;
            if (start + actual > train->n) actual = train->n - start;

            /* 4D input [actual, 1, 28, 28] */
            int shX[4] = {actual, 1, 28, 28};
            int shY[2] = {actual, 10};
            Tensor *X    = tensor_zeros(shX, 4);
            Tensor *Y    = tensor_zeros(shY, 2);
            Tensor *pred_b = tensor_zeros(shY, 2);
            Tensor *grad_b = tensor_zeros(shY, 2);

            memcpy(X->data,
                   train->images->data + start * 784,
                   (size_t)actual * 784 * sizeof(float));
            memcpy(Y->data,
                   train->labels->data + start * 10,
                   (size_t)actual * 10 * sizeof(float));

            float loss = nn_train_step(net, X, Y,
                                       LOSS_CROSS_ENTROPY,
                                       pred_b, grad_b);
            total_loss += loss;

            /* update all layers with learning rate */
            for (int i = 0; i < net->num_layers; i++) {
                Layer *l = &net->layers[i];
                l->update(l->layer_data, LR);
            }

            int preds[BATCH_SIZE], trues[BATCH_SIZE];
            tensor_argmax_rows(pred_b, preds);
            tensor_argmax_rows(Y,      trues);
            for (int i = 0; i < actual; i++)
                if (preds[i] == trues[i]) correct++;

            tensor_free(X); tensor_free(Y);
            tensor_free(pred_b); tensor_free(grad_b);
        }

        float train_loss = total_loss / n_batches;
        float train_acc  = (float)correct / train->n * 100.0f;
        float test_acc   = evaluate(net, test);

        printf("  %-6d %-12.4f %-12.1f%% %-10.1f%%\n",
               ep, train_loss, train_acc, test_acc);

        nn_save(net, "cnn_mnist_best.bin");
    }

    printf("\nModel saved: cnn_mnist_best.bin\n");

    tensor_free(pred); tensor_free(grad);
    adam_free(opt);
    nn_free(net);
    mnist_free(train);
    mnist_free(test);
    return 0;
}
