/*
 * demo_rnn.c — neuralc RNN & LSTM sine wave prediction demo
 *
 * Build: make rnn_demo
 * Run:   ./rnn_demo
 *
 * KEY RULE: always free reshape views AFTER backward, not before.
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "rnn.h"
#include "layer.h"

#ifdef SEQ_LEN
#undef SEQ_LEN
#endif
#define SEQ_LEN 10
#define BATCH    4
#define HIDDEN  16
#define RNN_EPOCHS  2000
#define RNN_LR      0.003f
#define LSTM_EPOCHS 3000
#define LSTM_LR     0.0005f

static void gen_sine(float *X, float *Y, int batch, int seq, float phase) {
    for (int b = 0; b < batch; b++)
        for (int t = 0; t < seq; t++) {
            X[b*seq+t] = sinf(phase + b*0.5f + t*0.4f);
            Y[b*seq+t] = sinf(phase + b*0.5f + (t+1)*0.4f);
        }
}

static float mse_loss(float *pred, float *tgt, float *grad, int n) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = pred[i] - tgt[i];
        loss += d*d;
        grad[i] = 2.0f*d/n;
    }
    return loss/n;
}

/* ── Vanilla RNN ──────────────────────────────────────────────────── */
static void demo_rnn(void) {
    printf("╔══════════════════════════════════╗\n");
    printf("║  Vanilla RNN — Sine Prediction   ║\n");
    printf("╚══════════════════════════════════╝\n");

    int N = BATCH*SEQ_LEN;
    float *Xd = calloc(N, sizeof(float));
    float *Yd = calloc(N, sizeof(float));

    int shX[3]={BATCH,SEQ_LEN,1};
    int shH[3]={BATCH,SEQ_LEN,HIDDEN};
    int shP[3]={BATCH,SEQ_LEN,1};
    int shFH[2]={N,HIDDEN};
    int shFP[2]={N,1};

    Tensor *X    = tensor_zeros(shX,3);
    Tensor *Hout = tensor_zeros(shH,3);
    Tensor *Pred = tensor_zeros(shP,3);
    Tensor *Grad = tensor_zeros(shP,3);

    RNNLayer   *rnn = rnn_create(1,HIDDEN);
    DenseLayer *fc  = dense_create(HIDDEN,1,ACT_NONE);
    printf("  RNN params:%d  Dense params:%d\n",
           rnn_param_count(rnn),dense_param_count(fc));

    for (int ep = 0; ep <= RNN_EPOCHS; ep++) {
        gen_sine(Xd, Yd, BATCH, SEQ_LEN, ep*0.05f);
        memcpy(X->data, Xd, N*sizeof(float));

        /* forward */
        rnn_forward(rnn,X,NULL,Hout);
        Tensor *Hf = tensor_reshape(Hout,shFH,2);
        Tensor *Pf = tensor_reshape(Pred,shFP,2);
        dense_forward(fc,Hf,Pf);
        tensor_free(Pf);

        /* loss */
        float loss = mse_loss(Pred->data,Yd,Grad->data,N);
        if (ep%400==0) printf("  Epoch %4d  loss=%.6f\n",ep,loss);

        /* backward — free Hf AFTER dense_backward */
        Tensor *Gf = tensor_reshape(Grad,shFP,2);
        dense_backward(fc,Gf);
        tensor_free(Gf);
        tensor_free(Hf);   /* safe now */

        Tensor *dHs = tensor_reshape(fc->dX,shH,3);
        rnn_backward(rnn,dHs);
        tensor_free(dHs);

        rnn_update_sgd(rnn,RNN_LR);
        for (size_t i=0;i<fc->W->size;i++) fc->W->data[i]-=RNN_LR*fc->dW->data[i];
        for (size_t i=0;i<fc->b->size;i++) fc->b->data[i]-=RNN_LR*fc->db->data[i];
    }

    /* show predictions */
    gen_sine(Xd,Yd,1,SEQ_LEN,0.0f);
    memcpy(X->data,Xd,SEQ_LEN*sizeof(float));
    int shX1[3]={1,SEQ_LEN,1}, shH1[3]={1,SEQ_LEN,HIDDEN};
    int shFH1[2]={SEQ_LEN,HIDDEN}, shFP1[2]={SEQ_LEN,1};
    int shP1[3]={1,SEQ_LEN,1};
    Tensor *X1   = tensor_zeros(shX1,3);
    Tensor *H1   = tensor_zeros(shH1,3);
    Tensor *P1   = tensor_zeros(shP1,3);
    memcpy(X1->data,Xd,SEQ_LEN*sizeof(float));
    rnn_forward(rnn,X1,NULL,H1);
    Tensor *Hf1 = tensor_reshape(H1,shFH1,2);
    Tensor *Pf1 = tensor_reshape(P1,shFP1,2);
    dense_forward(fc,Hf1,Pf1);
    tensor_free(Hf1); tensor_free(Pf1);

    printf("\n  t   input    pred     target\n");
    for (int t=0;t<SEQ_LEN;t++)
        printf("  %d  %6.3f   %6.3f   %6.3f\n",
               t,X1->data[t],P1->data[t],Yd[t]);

    tensor_free(X); tensor_free(Hout); tensor_free(Pred); tensor_free(Grad);
    tensor_free(X1); tensor_free(H1); tensor_free(P1);
    rnn_free(rnn); dense_free(fc);
    free(Xd); free(Yd);
}

/* ── LSTM ─────────────────────────────────────────────────────────── */
static void demo_lstm(void) {
    printf("\n╔══════════════════════════════════╗\n");
    printf("║  LSTM      — Sine Prediction     ║\n");
    printf("╚══════════════════════════════════╝\n");

    int N = BATCH*SEQ_LEN;
    float *Xd = calloc(N, sizeof(float));
    float *Yd = calloc(N, sizeof(float));

    int shX[3]={BATCH,SEQ_LEN,1};
    int shH[3]={BATCH,SEQ_LEN,HIDDEN};
    int shP[3]={BATCH,SEQ_LEN,1};
    int shFH[2]={N,HIDDEN};
    int shFP[2]={N,1};

    Tensor *X    = tensor_zeros(shX,3);
    Tensor *Hout = tensor_zeros(shH,3);
    Tensor *Pred = tensor_zeros(shP,3);
    Tensor *Grad = tensor_zeros(shP,3);

    LSTMLayer  *lstm = lstm_create(1,HIDDEN);
    DenseLayer *fc   = dense_create(HIDDEN,1,ACT_NONE);
    printf("  LSTM params:%d  Dense params:%d\n",
           lstm_param_count(lstm),dense_param_count(fc));

    gen_sine(Xd,Yd,BATCH,SEQ_LEN,0.0f);
    memcpy(X->data,Xd,N*sizeof(float));
    for (int ep = 0; ep <= LSTM_EPOCHS; ep++) {
        /* fixed dataset — stable training */
        lstm_forward(lstm,X,NULL,NULL,Hout);
        Tensor *Hf = tensor_reshape(Hout,shFH,2);
        Tensor *Pf = tensor_reshape(Pred,shFP,2);
        dense_forward(fc,Hf,Pf);
        tensor_free(Pf);

        float loss = mse_loss(Pred->data,Yd,Grad->data,N);
        if (ep%600==0) printf("  Epoch %4d  loss=%.6f\n",ep,loss);

        Tensor *Gf = tensor_reshape(Grad,shFP,2);
        dense_backward(fc,Gf);
        tensor_free(Gf);
        tensor_free(Hf);  /* free AFTER backward */

        Tensor *dHs = tensor_reshape(fc->dX,shH,3);
        lstm_backward(lstm,dHs);
        tensor_free(dHs);

        lstm_clip_gradients(lstm, 1.0f);
        lstm_update_sgd(lstm,LSTM_LR);
        for (size_t i=0;i<fc->W->size;i++) fc->W->data[i]-=LSTM_LR*fc->dW->data[i];
        for (size_t i=0;i<fc->b->size;i++) fc->b->data[i]-=LSTM_LR*fc->db->data[i];
    }

    gen_sine(Xd,Yd,1,SEQ_LEN,0.0f);
    int shX1[3]={1,SEQ_LEN,1}, shH1[3]={1,SEQ_LEN,HIDDEN};
    int shFH1[2]={SEQ_LEN,HIDDEN}, shFP1[2]={SEQ_LEN,1};
    int shP1[3]={1,SEQ_LEN,1};
    Tensor *X1 = tensor_zeros(shX1,3); memcpy(X1->data,Xd,SEQ_LEN*sizeof(float));
    Tensor *H1 = tensor_zeros(shH1,3);
    Tensor *P1 = tensor_zeros(shP1,3);
    lstm_forward(lstm,X1,NULL,NULL,H1);
    Tensor *Hf1 = tensor_reshape(H1,shFH1,2);
    Tensor *Pf1 = tensor_reshape(P1,shFP1,2);
    dense_forward(fc,Hf1,Pf1);
    tensor_free(Hf1); tensor_free(Pf1);

    printf("\n  t   input    pred     target\n");
    for (int t=0;t<SEQ_LEN;t++)
        printf("  %d  %6.3f   %6.3f   %6.3f\n",
               t,X1->data[t],P1->data[t],Yd[t]);

    tensor_free(X); tensor_free(Hout); tensor_free(Pred); tensor_free(Grad);
    tensor_free(X1); tensor_free(H1); tensor_free(P1);
    lstm_free(lstm); dense_free(fc);
    free(Xd); free(Yd);
}

int main(void){
    printf("╔════════════════════════════════════╗\n");
    printf("║   neuralc — RNN & LSTM Demo        ║\n");
    printf("╚════════════════════════════════════╝\n\n");
    demo_rnn();
    demo_lstm();
    printf("\n✓ RNN & LSTM complete!\n");
    return 0;
}
