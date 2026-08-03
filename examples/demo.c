/*
 * demo.c — Train a small network to solve XOR using cnet
 *
 *  Architecture:  2 → 8 (ReLU) → 1 (Sigmoid)
 *  Loss:          Binary Cross-Entropy
 *  Optimizer:     Adam (lr=0.01)
 *  Epochs:        5000
 */
#include <stdio.h>
#include "tensor.h"
#include "layer.h"
#include "nn.h"
#include "optimizer.h"

int main(void) {
    /* ── dataset: all 4 XOR combinations ── */
    float X_data[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float Y_data[4][1] = {{0},  {1},  {1},  {0}};

    int xsh[2] = {4, 2};
    int ysh[2] = {4, 1};

    Tensor *X      = tensor_zeros(xsh, 2);
    Tensor *Y      = tensor_zeros(ysh, 2);
    Tensor *pred   = tensor_zeros(ysh, 2);
    Tensor *grad   = tensor_zeros(ysh, 2);

    for (int i = 0; i < 4; i++) {
        X->data[i*2+0] = X_data[i][0];
        X->data[i*2+1] = X_data[i][1];
        Y->data[i]     = Y_data[i][0];
    }

    /* ── build network ── */
    Network *net = nn_create();
    nn_add_dense(net, dense_create(2, 8, ACT_RELU));
    nn_add_dense(net, dense_create(8, 1, ACT_SIGMOID));
    nn_print_summary(net);

    /* ── optimizer ── */
    Adam *opt = adam_create(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);

    /* ── training loop ── */
    int epochs = 5000;
    for (int ep = 0; ep <= epochs; ep++) {
        float loss = nn_train_step(net, X, Y, LOSS_BINARY_CROSS, pred, grad);
        adam_step(opt, net);

        if (ep % 500 == 0)
            printf("Epoch %4d  loss=%.6f\n", ep, loss);
    }

    /* ── evaluation ── */
    printf("\n--- XOR Predictions ---\n");
    nn_forward(net, X, pred, /*training=*/0);
    for (int i = 0; i < 4; i++)
        printf("  [%d XOR %d]  pred=%.4f  (expected %d)\n",
               (int)X->data[i*2], (int)X->data[i*2+1],
               pred->data[i], (int)Y->data[i]);

    /* ── save weights ── */
    if (nn_save(net, "xor_weights.bin") == 0)
        printf("\nWeights saved to xor_weights.bin\n");

    /* ── cleanup ── */
    tensor_free(X); tensor_free(Y);
    tensor_free(pred); tensor_free(grad);
    adam_free(opt);
    nn_free(net);

    return 0;
}
