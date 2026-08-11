/*
 * cuda/src/cuda_stub.c
 * ────────────────────────────────────────────────────────────────
 * CPU Fallback Stubs backing cuda_backend.h when GPU mode is disabled.
 */

#include "../include/cuda_backend.h"
#include <stdio.h>
#include <stdlib.h>

int    cuda_available(void)         { return 0; }
int    cuda_get_device_count(void) { return 0; }
void   cuda_set_device(int id)     { (void)id; }
int    cuda_get_device(void)       { return 0; }
void  *cuda_stream_create(void)    { return NULL; }
void   cuda_stream_destroy(void *s){ (void)s; }
void   cuda_set_stream(void *s)    { (void)s; }
void  *cuda_get_stream(void)       { return NULL; }
void   cuda_enable_tensor_cores(int e) { (void)e; }

void   cuda_pool_init(size_t b)    { (void)b; }
void  *cuda_pool_alloc(size_t b)   { (void)b; return NULL; }
void   cuda_pool_reset(void)       {}
void   cuda_pool_destroy(void)     {}
void   cuda_pool_info(size_t *c, size_t *u, size_t *p) { if (c) *c=0; if (u) *u=0; if (p) *p=0; }

size_t estimate_dense_vram_bytes(int batch, int in_f, int out_f) {
    size_t params = (size_t)out_f * in_f + out_f;
    size_t activations = (size_t)batch * (in_f + out_f);
    size_t gradients = params + activations;
    return (params + activations + gradients) * sizeof(float);
}

void cuda_get_vram_info(size_t *free_bytes, size_t *total_bytes) {
    if (free_bytes)  *free_bytes  = 0;
    if (total_bytes) *total_bytes = 0;
}

float *cuda_malloc_f32(size_t n)   { (void)n; return NULL; }
void   cuda_free_f32(float *p)     { (void)p; }
float *cuda_malloc_host(size_t n)  { (void)n; return NULL; }
void   cuda_free_host(float *p)    { (void)p; }

void   cuda_memcpy_h2d(float *d, const float *s, size_t n) { (void)d; (void)s; (void)n; }
void   cuda_memcpy_d2h(float *d, const float *s, size_t n) { (void)d; (void)s; (void)n; }
void   cuda_memcpy_d2d(float *d, const float *s, size_t n) { (void)d; (void)s; (void)n; }
void   cuda_memcpy_h2d_async(float *d, const float *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; }
void   cuda_memcpy_d2h_async(float *d, const float *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; }
void   cuda_memcpy_d2d_async(float *d, const float *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; }
void   cuda_memset_zero(float *p, size_t n) { (void)p; (void)n; }
void   cuda_sync(void)             {}

void cuda_matmul(const float *a, const float *b, float *out, int M, int K, int N) { (void)a; (void)b; (void)out; (void)M; (void)K; (void)N; }
void cuda_transpose(const float *a, float *out, int r, int c) { (void)a; (void)out; (void)r; (void)c; }
void cuda_linear_forward(const float *x, const float *W, const float *b, float *z, int batch, int i, int o) { (void)x; (void)W; (void)b; (void)z; (void)batch; (void)i; (void)o; }
void cuda_linear_backward(const float *x, const float *W, const float *dz, float *dW, float *db, float *dX, int batch, int i, int o) { (void)x; (void)W; (void)dz; (void)dW; (void)db; (void)dX; (void)batch; (void)i; (void)o; }
void cuda_fused_linear_relu(const float *x, const float *W, const float *b, float *out, int batch, int i, int o) { (void)x; (void)W; (void)b; (void)out; (void)batch; (void)i; (void)o; }
void cuda_fused_bias_add(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_bias_relu(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_bias_gelu(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_bias_silu(float *z, const float *b, int batch, int cols) { (void)z; (void)b; (void)batch; (void)cols; }
void cuda_fused_add_relu(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_fused_mul_relu(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_fused_add_bias_relu(const float *a, const float *b, float *out, int batch, int cols) { (void)a; (void)b; (void)out; (void)batch; (void)cols; }

void cuda_add(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_sub(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_mul(const float *a, const float *b, float *out, size_t n) { (void)a; (void)b; (void)out; (void)n; }
void cuda_scale(const float *a, float s, float *out, size_t n) { (void)a; (void)s; (void)out; (void)n; }
void cuda_add_scalar(const float *a, float s, float *out, size_t n) { (void)a; (void)s; (void)out; (void)n; }
void cuda_fill(float *a, float val, size_t n) { (void)a; (void)val; (void)n; }

void cuda_relu(const float *a, float *out, size_t n) { (void)a; (void)out; (void)n; }
void cuda_relu_grad(const float *a, const float *g, float *out, size_t n) { (void)a; (void)g; (void)out; (void)n; }
void cuda_sigmoid(const float *a, float *out, size_t n) { (void)a; (void)out; (void)n; }
void cuda_sigmoid_grad(const float *s, const float *g, float *out, size_t n) { (void)s; (void)g; (void)out; (void)n; }
void cuda_tanh_f(const float *a, float *out, size_t n) { (void)a; (void)out; (void)n; }
void cuda_tanh_grad(const float *t, const float *g, float *out, size_t n) { (void)t; (void)g; (void)out; (void)n; }
void cuda_softmax_rows(const float *a, float *out, int r, int c) { (void)a; (void)out; (void)r; (void)c; }

void cuda_pad2d(const float *in, float *out, int b, int C, int H, int W, int pad) { (void)in; (void)out; (void)b; (void)C; (void)H; (void)W; (void)pad; }
void cuda_unpad2d(const float *p, float *out, int b, int C, int H, int W, int pad) { (void)p; (void)out; (void)b; (void)C; (void)H; (void)W; (void)pad; }
void cuda_conv2d_forward(const float *in, const float *W, const float *b, float *out, int b_n, int iC, int pH, int pW, int oC, int kH, int kW, int st, int oH, int oW) { (void)in; (void)W; (void)b; (void)out; (void)b_n; (void)iC; (void)pH; (void)pW; (void)oC; (void)kH; (void)kW; (void)st; (void)oH; (void)oW; }
void cuda_conv2d_backward(const float *in, const float *W, const float *g_out, float *dW, float *db, float *dp, int b_n, int iC, int pH, int pW, int oC, int kH, int kW, int st, int oH, int oW) { (void)in; (void)W; (void)g_out; (void)dW; (void)db; (void)dp; (void)b_n; (void)iC; (void)pH; (void)pW; (void)oC; (void)kH; (void)kW; (void)st; (void)oH; (void)oW; }

void cuda_maxpool2d_forward(const float *in, float *out, float *mask, int b, int C, int H, int W, int ps, int st, int oH, int oW) { (void)in; (void)out; (void)mask; (void)b; (void)C; (void)H; (void)W; (void)ps; (void)st; (void)oH; (void)oW; }
void cuda_maxpool2d_backward(const float *go, float *gi, const float *mask, int b, int C, int H, int W, int ps, int st, int oH, int oW) { (void)go; (void)gi; (void)mask; (void)b; (void)C; (void)H; (void)W; (void)ps; (void)st; (void)oH; (void)oW; }

void *cuda_malloc_f16(size_t n) { (void)n; return NULL; }
void  cuda_free_f16(void *p) { (void)p; }
void  cuda_f32_to_f16(const float *s, void *d, size_t n) { (void)s; (void)d; (void)n; }
void  cuda_f16_to_f32(const void *s, float *d, size_t n) { (void)s; (void)d; (void)n; }
void  cuda_matmul_fp16(const void *a, const void *b, void *o, int M, int K, int N) { (void)a; (void)b; (void)o; (void)M; (void)K; (void)N; }
void  cuda_linear_forward_fp16(const void *x, const void *w, const float *b, void *z, int ba, int i, int o) { (void)x; (void)w; (void)b; (void)z; (void)ba; (void)i; (void)o; }
void  cuda_linear_backward_fp16(const void *x, const void *w, const void *dz, void *dw, float *db, void *dx, int ba, int i, int o) { (void)x; (void)w; (void)dz; (void)dw; (void)db; (void)dx; (void)ba; (void)i; (void)o; }
