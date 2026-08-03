/*
 * main.c — neuralc full feature demonstration
 *
 * Shows all major features of the neuralc library:
 *   1. Tensor operations
 *   2. Dense network (XOR problem)
 *   3. Dropout layer
 *   4. Batch Normalization
 *   5. Conv2D + MaxPool + Flatten
 *   6. All optimizers (SGD, Adam, RMSProp)
 *   7. Save and load weights
 *
 * Build:  make
 * Run:    ./neuralc
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"
#include "layer.h"
#include "nn.h"
#include "optimizer.h"
#include "dropout.h"
#include "batchnorm.h"
#include "conv.h"

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 1 — Tensor operations demo
 * ═══════════════════════════════════════════════════════════════════ */
static void demo_tensors(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  1. Tensor Operations        ║\n");
    printf("╚══════════════════════════════╝\n");

    /* create tensors */
    int sh[2] = {3, 3};
    Tensor *a = tensor_rand(sh, 2);
    Tensor *b = tensor_rand(sh, 2);
    Tensor *c = tensor_zeros(sh, 2);

    tensor_print(a, "a (random)");
    tensor_print(b, "b (random)");

    /* element-wise ops */
    tensor_add(a, b, c);   tensor_print(c, "a + b");
    tensor_mul(a, b, c);   tensor_print(c, "a * b (hadamard)");
    tensor_scale(a, 2.0f, c); tensor_print(c, "a * 2.0");

    /* reductions */
    printf("  sum(a)  = %.4f\n", tensor_sum(a));
    printf("  mean(a) = %.4f\n", tensor_mean(a));
    printf("  max(a)  = %.4f\n", tensor_max(a));
    printf("  argmax  = %d\n",   tensor_argmax(a));

    /* matrix multiply */
    int shA[2] = {2, 3}, shB[2] = {3, 2}, shC[2] = {2, 2};
    Tensor *A  = tensor_rand(shA, 2);
    Tensor *B  = tensor_rand(shB, 2);
    Tensor *C  = tensor_zeros(shC, 2);
    tensor_matmul(A, B, C);
    tensor_print(A, "A [2x3]");
    tensor_print(B, "B [3x2]");
    tensor_print(C, "A @ B [2x2]");

    /* activations */
    int sh1[2] = {1, 5};
    Tensor *x   = tensor_rand(sh1, 2);
    Tensor *out = tensor_zeros(sh1, 2);
    tensor_relu(x, out);    tensor_print(out, "relu(x)");
    tensor_sigmoid(x, out); tensor_print(out, "sigmoid(x)");
    tensor_softmax(x, out); tensor_print(out, "softmax(x)");

    /* axis operations */
    int sh2[2] = {3, 4};
    Tensor *m    = tensor_rand(sh2, 2);
    int    shcol[1] = {4};
    int    shrow[1] = {3};
    Tensor *col_sum = tensor_zeros(shcol, 1);
    Tensor *row_sum = tensor_zeros(shrow, 1);
    tensor_sum_axis(m, 0, col_sum); tensor_print(col_sum, "sum axis=0 (col sums)");
    tensor_sum_axis(m, 1, row_sum); tensor_print(row_sum, "sum axis=1 (row sums)");

    tensor_free(a); tensor_free(b); tensor_free(c);
    tensor_free(A); tensor_free(B); tensor_free(C);
    tensor_free(x); tensor_free(out);
    tensor_free(m); tensor_free(col_sum); tensor_free(row_sum);
}

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 2 — XOR with Dense + Adam
 * ═══════════════════════════════════════════════════════════════════ */
static void demo_xor(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  2. XOR — Dense + Adam       ║\n");
    printf("╚══════════════════════════════╝\n");

    float X_d[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float Y_d[4][1] = {{0},  {1},  {1},  {0}};
    int xsh[2]={4,2}, ysh[2]={4,1};

    Tensor *X    = tensor_zeros(xsh, 2);
    Tensor *Y    = tensor_zeros(ysh, 2);
    Tensor *pred = tensor_zeros(ysh, 2);
    Tensor *grad = tensor_zeros(ysh, 2);
    for (int i = 0; i < 4; i++) {
        X->data[i*2]   = X_d[i][0];
        X->data[i*2+1] = X_d[i][1];
        Y->data[i]     = Y_d[i][0];
    }

    Network *net = nn_create();
    nn_add_dense(net, dense_create(2, 16, ACT_RELU));
    nn_add_dense(net, dense_create(16, 1, ACT_SIGMOID));
    nn_print_summary(net);

    Adam *opt = adam_create(0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);

    for (int ep = 0; ep <= 3000; ep++) {
        float loss = nn_train_step(net, X, Y, LOSS_BINARY_CROSS, pred, grad);
        adam_step(opt, net);
        if (ep % 500 == 0)
            printf("  Epoch %4d  loss=%.6f\n", ep, loss);
    }

    nn_forward(net, X, pred, /*training=*/0);
    float acc = nn_accuracy_binary(pred, Y);
    printf("  Accuracy: %.1f%%\n", acc * 100.0f);
    printf("  Predictions:\n");
    for (int i = 0; i < 4; i++)
        printf("    [%d XOR %d] = %.4f  (expected %d)\n",
               (int)X->data[i*2], (int)X->data[i*2+1],
               pred->data[i], (int)Y->data[i]);

    /* save weights */
    nn_save(net, "xor_weights.bin");
    printf("  Weights saved → xor_weights.bin\n");

    tensor_free(X); tensor_free(Y);
    tensor_free(pred); tensor_free(grad);
    adam_free(opt);
    nn_free(net);
}

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 3 — Dropout demo
 * ═══════════════════════════════════════════════════════════════════ */
static void demo_dropout(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  3. Dropout Layer            ║\n");
    printf("╚══════════════════════════════╝\n");

    int sh[2] = {1, 8};
    Tensor *x   = tensor_ones(sh, 2);
    Tensor *out = tensor_zeros(sh, 2);

    DropoutLayer *drop = dropout_create(0.5f);

    /* training mode: ~50% neurons zeroed */
    dropout_train(drop);
    dropout_forward(drop, x, out);
    tensor_print(out, "dropout train (50% drop)");

    /* eval mode: pass through unchanged */
    dropout_eval(drop);
    dropout_forward(drop, x, out);
    tensor_print(out, "dropout eval  (no drop)");

    tensor_free(x); tensor_free(out);
    dropout_free(drop);
}

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 4 — Batch Normalization demo
 * ═══════════════════════════════════════════════════════════════════ */
static void demo_batchnorm(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  4. Batch Normalization      ║\n");
    printf("╚══════════════════════════════╝\n");

    /* input: 4 samples, 3 features with large values */
    int sh[2] = {4, 3};
    Tensor *x   = tensor_zeros(sh, 2);
    Tensor *out = tensor_zeros(sh, 2);

    float vals[12] = {100, 200, 300,
                      400, 500, 600,
                      700, 800, 900,
                      1000,1100,1200};
    memcpy(x->data, vals, 12*sizeof(float));

    BatchNorm *bn = batchnorm_create(3, 1e-5f, 0.1f);
    batchnorm_train(bn);
    batchnorm_forward(bn, x, out);

    tensor_print(x,   "input  (large values)");
    tensor_print(out, "output (normalized)");

    tensor_free(x); tensor_free(out);
    batchnorm_free(bn);
}

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 5 — Conv2D + MaxPool + Flatten demo
 * ═══════════════════════════════════════════════════════════════════ */
static void demo_conv2d(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  5. Conv2D + MaxPool         ║\n");
    printf("╚══════════════════════════════╝\n");

    /* simulate a tiny batch of 2 grayscale 8x8 images */
    int sh_in[4] = {2, 1, 8, 8};
    Tensor *input = tensor_rand(sh_in, 4);
    printf("  Input:  [batch=2, ch=1, H=8, W=8]\n");

    /* Conv2D: 1→4 channels, 3x3 kernel, stride=1, pad=1 (same) */
    Conv2D *conv = conv2d_create(1, 4, 3, 1, 1);
    int out_H, out_W;
    conv2d_output_size(conv, 8, 8, &out_H, &out_W);
    printf("  Conv2D(1→4, 3x3, pad=1): output [2,4,%d,%d]  params=%d\n",
           out_H, out_W, conv2d_param_count(conv));

    int sh_conv[4] = {2, 4, out_H, out_W};
    Tensor *conv_out = tensor_zeros(sh_conv, 4);
    conv2d_forward(conv, input, conv_out);

    /* MaxPool 2x2 stride 2 */
    int pool_H = out_H / 2, pool_W = out_W / 2;
    int sh_pool[4] = {2, 4, pool_H, pool_W};
    Tensor *pool_out = tensor_zeros(sh_pool, 4);
    Tensor *pool_mask= tensor_zeros(sh_pool, 4);
    maxpool2d_forward(conv_out, pool_out, pool_mask, 2, 2);
    printf("  MaxPool(2x2):            output [2,4,%d,%d]\n",
           pool_H, pool_W);

    /* Flatten for Dense layer */
    Tensor *flat = flatten(pool_out);
    printf("  Flatten:                 output [2,%d]\n", flat->shape[1]);

    /* Dense classifier */
    DenseLayer *fc = dense_create(flat->shape[1], 10, ACT_SOFTMAX);
    int sh_fc[2] = {2, 10};
    Tensor *fc_out = tensor_zeros(sh_fc, 2);
    dense_forward(fc, flat, fc_out);
    printf("  Dense(%d→10, softmax):  output [2,10]\n", flat->shape[1]);
    tensor_print(fc_out, "classifier output (softmax probs)");

    /* argmax → predicted class */
    int preds[2];
    tensor_argmax_rows(fc_out, preds);
    printf("  Predicted classes: [%d, %d]\n", preds[0], preds[1]);

    tensor_free(input);
    tensor_free(conv_out);
    tensor_free(pool_out);
    tensor_free(pool_mask);
    tensor_free(flat);
    tensor_free(fc_out);
    conv2d_free(conv);
    dense_free(fc);
}

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 6 — Optimizer comparison
 * ═══════════════════════════════════════════════════════════════════ */
static float train_xor_with(const char *name, Network *net,
                             int epochs, float lr) {
    float X_d[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float Y_d[4][1] = {{0},{1},{1},{0}};
    int xsh[2]={4,2}, ysh[2]={4,1};
    Tensor *X    = tensor_zeros(xsh,2);
    Tensor *Y    = tensor_zeros(ysh,2);
    Tensor *pred = tensor_zeros(ysh,2);
    Tensor *grad = tensor_zeros(ysh,2);
    for (int i=0;i<4;i++) {
        X->data[i*2]=X_d[i][0]; X->data[i*2+1]=X_d[i][1];
        Y->data[i]=Y_d[i][0];
    }

    float final_loss = 0.0f;
    if (strcmp(name,"SGD")==0) {
        SGD *opt = sgd_create(lr, 0.9f, 0.0f);
        for (int ep=0;ep<epochs;ep++) {
            final_loss = nn_train_step(net,X,Y,LOSS_BINARY_CROSS,pred,grad);
            sgd_step(opt,net);
        }
        sgd_free(opt);
    } else if (strcmp(name,"Adam")==0) {
        Adam *opt = adam_create(lr,0.9f,0.999f,1e-8f,0.0f);
        for (int ep=0;ep<epochs;ep++) {
            final_loss = nn_train_step(net,X,Y,LOSS_BINARY_CROSS,pred,grad);
            adam_step(opt,net);
        }
        adam_free(opt);
    } else {
        RMSProp *opt = rmsprop_create(lr,0.9f,1e-8f,0.0f);
        for (int ep=0;ep<epochs;ep++) {
            final_loss = nn_train_step(net,X,Y,LOSS_BINARY_CROSS,pred,grad);
            rmsprop_step(opt,net);
        }
        rmsprop_free(opt);
    }

    tensor_free(X); tensor_free(Y);
    tensor_free(pred); tensor_free(grad);
    return final_loss;
}

static void demo_optimizers(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  6. Optimizer Comparison     ║\n");
    printf("╚══════════════════════════════╝\n");

    const char *names[] = {"SGD", "Adam", "RMSProp"};
    float       lrs[]   = {0.05f, 0.01f, 0.01f};

    for (int i = 0; i < 3; i++) {
        Network *net = nn_create();
        nn_add_dense(net, dense_create(2, 8, ACT_RELU));
        nn_add_dense(net, dense_create(8, 1, ACT_SIGMOID));
        float loss = train_xor_with(names[i], net, 2000, lrs[i]);
        printf("  %-10s lr=%.4f  final_loss=%.6f\n",
               names[i], lrs[i], loss);
        nn_free(net);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  SECTION 7 — Save and reload weights
 * ═══════════════════════════════════════════════════════════════════ */
static void demo_save_load(void) {
    printf("\n╔══════════════════════════════╗\n");
    printf("║  7. Save & Load Weights      ║\n");
    printf("╚══════════════════════════════╝\n");

    /* reload XOR weights saved in section 2 */
    Network *net = nn_create();
    nn_add_dense(net, dense_create(2, 16, ACT_RELU));
    nn_add_dense(net, dense_create(16,  1, ACT_SIGMOID));

    if (nn_load(net, "xor_weights.bin") == 0) {
        printf("  Loaded xor_weights.bin\n");
        float X_d[4][2] = {{0,0},{0,1},{1,0},{1,1}};
        int xsh[2]={4,2}, ysh[2]={4,1};
        Tensor *X    = tensor_zeros(xsh,2);
        Tensor *pred = tensor_zeros(ysh,2);
        for (int i=0;i<4;i++) {
            X->data[i*2]=X_d[i][0];
            X->data[i*2+1]=X_d[i][1];
        }
        nn_forward(net, X, pred, /*training=*/0);
        printf("  Reloaded predictions:\n");
        for (int i=0;i<4;i++)
            printf("    [%d XOR %d] = %.4f\n",
                   (int)X->data[i*2],(int)X->data[i*2+1],pred->data[i]);
        tensor_free(X); tensor_free(pred);
    } else {
        printf("  (no saved weights found — run section 2 first)\n");
    }
    nn_free(net);
}

/* ═══════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("╔════════════════════════════════════╗\n");
    printf("║   neuralc — Full Feature Demo      ║\n");
    printf("║   Deep learning in pure C          ║\n");
    printf("╚════════════════════════════════════╝\n");

    demo_tensors();
    demo_xor();
    demo_dropout();
    demo_batchnorm();
    demo_conv2d();
    demo_optimizers();
    demo_save_load();

    printf("\n✓ All demos completed successfully!\n\n");
    return 0;
}
