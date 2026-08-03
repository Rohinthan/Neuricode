/*
 * kernels.cl — neuralc OpenCL GPU kernels
 *
 * Each kernel runs thousands of threads in parallel on the GPU.
 * Work items replace the inner loop of CPU ops.
 */

/* ── element-wise ops ─────────────────────────────────────────────── */

__kernel void tensor_add_gpu(
    __global const float *a,
    __global const float *b,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}

__kernel void tensor_sub_gpu(
    __global const float *a,
    __global const float *b,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] - b[i];
}

__kernel void tensor_mul_gpu(
    __global const float *a,
    __global const float *b,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] * b[i];
}

__kernel void tensor_scale_gpu(
    __global const float *a,
    const float s,
    __global float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] * s;
}

/* ── activations ──────────────────────────────────────────────────── */

__kernel void relu_gpu(
    __global const float *in,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__kernel void relu_grad_gpu(
    __global const float *in,
    __global const float *grad,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = in[i] > 0.0f ? grad[i] : 0.0f;
}

__kernel void sigmoid_gpu(
    __global const float *in,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = 1.0f / (1.0f + exp(-in[i]));
}

__kernel void sigmoid_grad_gpu(
    __global const float *sig,
    __global const float *grad,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float s = sig[i];
        out[i] = grad[i] * s * (1.0f - s);
    }
}

__kernel void tanh_gpu(
    __global const float *in,
    __global       float *out,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) out[i] = tanh(in[i]);
}

/* ── matrix multiply ──────────────────────────────────────────────── */
/*
 * Each work item computes one output element out[row, col]
 * a: [M, K]   b: [K, N]   out: [M, N]
 */
__kernel void matmul_gpu(
    __global const float *a,
    __global const float *b,
    __global       float *out,
    const int M,
    const int K,
    const int N)
{
    int row = get_global_id(0);   /* 0..M-1 */
    int col = get_global_id(1);   /* 0..N-1 */

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += a[row * K + k] * b[k * N + col];
        out[row * N + col] = acc;
    }
}

/* ── tiled matmul (faster — uses local memory) ───────────────────── */
#define TILE 16

__kernel void matmul_tiled_gpu(
    __global const float *a,
    __global const float *b,
    __global       float *out,
    const int M,
    const int K,
    const int N)
{
    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    int row   = get_global_id(0);
    int col   = get_global_id(1);
    int lrow  = get_local_id(0);
    int lcol  = get_local_id(1);

    float acc = 0.0f;
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        /* load tile from A */
        int aCol = t * TILE + lcol;
        tileA[lrow][lcol] = (row < M && aCol < K)
                             ? a[row * K + aCol] : 0.0f;
        /* load tile from B */
        int bRow = t * TILE + lrow;
        tileB[lrow][lcol] = (bRow < K && col < N)
                             ? b[bRow * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; k++)
            acc += tileA[lrow][k] * tileB[k][lcol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N)
        out[row * N + col] = acc;
}

/* ── dense layer forward ─────────────────────────────────────────── */
/*
 * Computes: out[b, o] = sum_i(input[b,i] * W[o,i]) + b[o]
 * Each work item handles one (batch, out_neuron) pair
 */
__kernel void dense_forward_gpu(
    __global const float *input,    /* [batch, in_f]  */
    __global const float *W,        /* [out_f, in_f]  */
    __global const float *bias,     /* [out_f]        */
    __global       float *output,   /* [batch, out_f] */
    const int batch,
    const int in_f,
    const int out_f)
{
    int b = get_global_id(0);   /* batch index  */
    int o = get_global_id(1);   /* output index */

    if (b < batch && o < out_f) {
        float acc = bias[o];
        for (int i = 0; i < in_f; i++)
            acc += input[b * in_f + i] * W[o * in_f + i];
        output[b * out_f + o] = acc;
    }
}

/* ── softmax (row-wise) ──────────────────────────────────────────── */
/*
 * One work item per row (batch element)
 * Not fully parallel but correct for small class counts
 */
__kernel void softmax_gpu(
    __global const float *in,
    __global       float *out,
    const int rows,
    const int cols)
{
    int r = get_global_id(0);
    if (r >= rows) return;

    /* find max for numerical stability */
    float mx = in[r * cols];
    for (int c = 1; c < cols; c++)
        if (in[r * cols + c] > mx) mx = in[r * cols + c];

    /* exp and sum */
    float s = 0.0f;
    for (int c = 0; c < cols; c++) {
        out[r * cols + c] = exp(in[r * cols + c] - mx);
        s += out[r * cols + c];
    }
    /* normalize */
    for (int c = 0; c < cols; c++)
        out[r * cols + c] /= s;
}
