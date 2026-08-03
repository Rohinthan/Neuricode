#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

/* Maximum number of dimensions supported */
#define TENSOR_MAX_DIMS 8

/* ── device residency ─────────────────────────────────────────────
 * CF_DEVICE_CPU: t->data is a host pointer, all ops run as before.
 * CF_DEVICE_GPU: t->data is a CUDA device pointer. Do NOT dereference
 *                it from host code (T1/T2 macros, tensor_get, etc.
 *                are only valid on CPU tensors) — copy back with
 *                tensor_to_cpu() first.
 * Built WITHOUT -DUSE_CUDA, tensor_to_gpu()/tensor_to_cpu() are still
 * declared but calling them is a hard error (cforge_error), so
 * existing CPU-only code and builds are completely unaffected.    */
typedef enum {
    CF_DEVICE_CPU = 0,
    CF_DEVICE_GPU = 1
} CF_Device;

/* Which GPU backend a CF_DEVICE_GPU tensor's `data` pointer belongs
 * to. Meaningless when device == CF_DEVICE_CPU.
 *   CF_GPU_CUDA:   data is a raw CUDA device pointer (cudaMalloc'd)
 *   CF_GPU_OPENCL: data is a `cl_mem` handle, reinterpret-cast to
 *                  float* (cl_mem is itself just an opaque pointer,
 *                  so this is a same-size, valid cast — see
 *                  opencl_backend.c) */
typedef enum {
    CF_GPU_NONE   = 0,
    CF_GPU_CUDA   = 1,
    CF_GPU_OPENCL = 2
} CF_GpuBackend;

/* Tensor and GraphNode are mutually referential (a Tensor points at the
 * node that produced it; a node points at its input Tensors), so both
 * tags are forward-declared before either body is defined. */
typedef struct Tensor    Tensor;
typedef struct GraphNode GraphNode;

/* ── autograd: computational graph tape node ──────────────────────
 * Every tracked op (one whose inputs have requires_grad == 1) pushes
 * one of these onto the global tape and stamps it into `output->node`.
 * A Tensor with node == NULL is a *graph leaf* — either a fresh
 * parameter/input that was never produced by a tracked op, or a
 * tensor whose graph was already released via tensor_tape_clear().
 * tensor_backward() walks the tape backwards from the loss, calling
 * each node's backward_fn exactly once in reverse topological order.
 */
struct GraphNode {
    Tensor  *output;              /* the tensor this node produced          */
    Tensor **parents;             /* dynamic array: the op's input tensors  *
                                    * (structural dependencies) — a parent   *
                                    * with parent->node == NULL is a leaf,   *
                                    * so recursion stops and the gradient is *
                                    * deposited straight into its ->grad     */
    int      n_parents;

    /* Isolated backward callback: recomputes and accumulates the local
     * gradient(s) for this op's parent(s) from node->output->grad.
     * Kept as a plain function pointer (rather than inline logic) so
     * every op — elementwise, matmul, permute, future ops — can plug
     * its own backward rule into one uniform tape entry. */
    void (*backward_fn)(GraphNode *node);

    /* Small generic integer scratchpad for whatever shape metadata the
     * backward_fn needs to reconstruct the local gradient (e.g. the
     * [M,K,N] dims for matmul, or the axis_order permutation). Sized
     * to TENSOR_MAX_DIMS so it comfortably covers both cases without
     * a separate malloc. */
    int ints[TENSOR_MAX_DIMS];

    /* Topological-sort marker: 0 = unvisited, 1 = on the recursion
     * stack (used to detect cycles), 2 = finished/ordered. */
    int visited;
};

struct Tensor {
    float  *data;           /* flat row-major data buffer          */
    int     shape[TENSOR_MAX_DIMS];
    int     ndim;           /* number of dimensions                */
    size_t  size;           /* total number of elements            */
    int     owns_data;      /* 1 = free data on tensor_free()      */
    CF_Device     device;      /* CF_DEVICE_CPU or CF_DEVICE_GPU   */
    CF_GpuBackend gpu_backend;  /* which GPU backend, if any       */

    /* ── autograd bookkeeping ─────────────────────────────────────
     * requires_grad : opt-in flag. Set via tensor_requires_grad_().
     *                 Any op whose inputs include a requires_grad
     *                 tensor tapes itself and marks its output the
     *                 same way, so the flag propagates forward
     *                 automatically through a computation.
     * grad          : lazily allocated (calloc'd on first use), same
     *                 length as `data`. Accumulated into (+=), never
     *                 overwritten, so multi-use tensors (fan-out in
     *                 the graph) correctly sum incoming gradients.
     * node          : GraphNode that produced this tensor, or NULL
     *                 for a leaf (see GraphNode comment above).
     * view_base     : for a non-owning view (owns_data == 0, e.g. a
     *                 tensor_reshape() result), the tensor that
     *                 actually owns the backing allocation. Views
     *                 never allocate their own `grad` buffer — every
     *                 gradient aimed at a view is redirected to
     *                 view_base->grad instead, using the view's own
     *                 shape metadata (valid because reshape/flatten
     *                 never reorders elements, only reinterprets the
     *                 shape, so the flat index space is identical
     *                 between a view and its base). NULL when
     *                 owns_data == 1 (the tensor is its own base).  */
    int        requires_grad;
    float     *grad;
    GraphNode *node;
    Tensor    *view_base;
};

/* ── lifecycle ─────────────────────────────────────────────────── */
Tensor *tensor_create(const int *shape, int ndim);
Tensor *tensor_zeros(const int *shape, int ndim);
Tensor *tensor_ones(const int *shape, int ndim);
Tensor *tensor_rand(const int *shape, int ndim);   /* uniform [0,1) */
Tensor *tensor_randn(const int *shape, int ndim);  /* N(0,1)        */
Tensor *tensor_clone(const Tensor *t);
void    tensor_free(Tensor *t);

/* ── element access ────────────────────────────────────────────── */
/* E.g. tensor_at2(t, i, j)  →  t->data[i*cols + j]               */
float  tensor_get(const Tensor *t, const int *idx);
void   tensor_set(Tensor *t, const int *idx, float val);

/* Convenience macros for 1-D / 2-D access */
#define T1(t, i)      ((t)->data[(i)])
#define T2(t, i, j)   ((t)->data[(i)*(t)->shape[1] + (j)])

/* ── element-wise ops (result written to out; out may == a or b) ─ */
void tensor_add(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_sub(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_mul(const Tensor *a, const Tensor *b, Tensor *out);  /* Hadamard */
void tensor_div(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_scale(const Tensor *a, float s, Tensor *out);
void tensor_add_scalar(const Tensor *a, float s, Tensor *out);
void tensor_neg(const Tensor *a, Tensor *out);
void tensor_abs(const Tensor *a, Tensor *out);
void tensor_square(const Tensor *a, Tensor *out);
void tensor_sqrt_t(const Tensor *a, Tensor *out);
void tensor_exp(const Tensor *a, Tensor *out);
void tensor_log_t(const Tensor *a, Tensor *out);
void tensor_clip(const Tensor *a, float lo, float hi, Tensor *out);

/* ── reductions ────────────────────────────────────────────────── */
float tensor_sum(const Tensor *t);
float tensor_mean(const Tensor *t);
float tensor_max(const Tensor *t);
float tensor_min(const Tensor *t);
float tensor_norm(const Tensor *t);   /* L2 norm */

/* ── argmax / argmin ────────────────────────────────────────────── */
/* tensor_argmax: index of max value across entire tensor           */
int   tensor_argmax(const Tensor *t);
/* tensor_argmin: index of min value across entire tensor           */
int   tensor_argmin(const Tensor *t);
/* tensor_argmax_rows: for 2D [batch,classes] → fills out[batch]    *
 *  with the argmax class index for each row                        */
void  tensor_argmax_rows(const Tensor *t, int *out);

/* ── axis reductions ────────────────────────────────────────────── */
/* tensor_sum_axis: sum along axis for 2D tensors                   *
 *   axis=0 → sum rows    → out shape [1, cols]                     *
 *   axis=1 → sum cols    → out shape [rows, 1]                     */
void  tensor_sum_axis(const Tensor *t, int axis, Tensor *out);
void  tensor_mean_axis(const Tensor *t, int axis, Tensor *out);

/* ── matrix ops (2-D tensors) ──────────────────────────────────── */
void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_transpose(const Tensor *a, Tensor *out);  /* 2-D only   */

/* ── activation functions (element-wise, in-place safe) ────────── */
void tensor_relu(const Tensor *a, Tensor *out);
void tensor_relu_grad(const Tensor *a, const Tensor *grad, Tensor *out);
void tensor_sigmoid(const Tensor *a, Tensor *out);
void tensor_sigmoid_grad(const Tensor *sig, const Tensor *grad, Tensor *out);
void tensor_tanh_t(const Tensor *a, Tensor *out);
void tensor_tanh_grad(const Tensor *th, const Tensor *grad, Tensor *out);
void tensor_softmax(const Tensor *a, Tensor *out);  /* row-wise for 2-D */

/* ── shape ops ──────────────────────────────────────────────────── */
Tensor *tensor_reshape(Tensor *t, const int *new_shape, int new_ndim);
void    tensor_fill(Tensor *t, float val);
void    tensor_copy_data(Tensor *dst, const Tensor *src);

/* ── layout transforms ────────────────────────────────────────────
 * tensor_permute: general N-D axis permutation (e.g. swap
 *   [Batch, Sequence, Features] -> [Sequence, Batch, Features] via
 *   axis_order = {1, 0, 2}). Materializes a fresh, contiguous,
 *   owning tensor with the new shape/layout (this is a real data
 *   copy, not a zero-copy stride view — required so downstream ops
 *   that assume simple row-major contiguity keep working unmodified).
 *   axis_order must be a permutation of [0 .. ndim-1] and ndim must
 *   equal src->ndim; both are validated with CF_CHECK. CPU only —
 *   call tensor_to_cpu() first if src lives on a GPU.
 * tensor_permute_backward: scatters grad_out (shaped like the
 *   permuted output) back into grad_in (shaped like the original,
 *   pre-permute tensor) using the inverse axis mapping. Accumulates
 *   (+=) rather than overwrites, so it composes safely with other
 *   contributions already sitting in grad_in. Exposed standalone (in
 *   addition to being wired into the autograd tape automatically by
 *   tensor_permute) so it can also be used to hand-roll custom
 *   backward passes. */
Tensor *tensor_permute(Tensor *src, const int *axis_order, int ndim);
void    tensor_permute_backward(const Tensor *grad_out, const int *axis_order, Tensor *grad_in);

/* ── autograd (dynamic, tape-based) ───────────────────────────────
 * tensor_requires_grad_: opt a leaf tensor (parameter/input) into
 *   gradient tracking. Trailing underscore marks it as in-place,
 *   PyTorch-style. Everything computed from a requires_grad tensor
 *   automatically requires_grad too and gets taped.
 * tensor_backward: runs reverse-mode autograd from a scalar `loss`
 *   tensor (size == 1) back through every tracked op that produced
 *   it, topologically sorting the tape and invoking each node's
 *   backward_fn exactly once, accumulating into each leaf's ->grad.
 * tensor_zero_grad: zero out (without freeing) a tensor's gradient
 *   buffer — call between training steps, before the next backward.
 * tensor_tape_clear: frees every GraphNode recorded so far and
 *   detaches them from their output tensors (output->node is reset
 *   to NULL), releasing the whole computation graph. Call this once
 *   you're done with a backward pass and don't need to run it again
 *   (retaining the graph across multiple backward() calls without
 *   clearing is not supported by this engine). */
void  tensor_requires_grad_(Tensor *t, int requires_grad);
void  tensor_backward(Tensor *loss);
void  tensor_zero_grad(Tensor *t);
void  tensor_tape_clear(void);

/* backend-check helpers — safe even when a backend isn't compiled in,
 * since gpu_backend is always a valid field on Tensor. Public so
 * conv.c/layer.c can guard against feeding a tensor to a kernel path
 * (e.g. Conv2D) that doesn't support its backend yet. */
#define CF_IS_CUDA(t)   ((t)->device == CF_DEVICE_GPU && (t)->gpu_backend == CF_GPU_CUDA)
#define CF_IS_OPENCL(t) ((t)->device == CF_DEVICE_GPU && (t)->gpu_backend == CF_GPU_OPENCL)

/* ── GPU residency (see cuda_backend.h / opencl_backend.h) ──────────
 * tensor_to_gpu:     auto-picks a backend — CUDA if this build has
 *                    -DUSE_CUDA and a device is available, else
 *                    OpenCL if -DUSE_OPENCL and a device is
 *                    available. In-place; returns t for chaining.
 *                    No-op if already on GPU.
 * tensor_to_gpu_ex:  same, but forces a specific backend — useful
 *                    when both USE_CUDA and USE_OPENCL are compiled
 *                    in and you want to pick explicitly (e.g. to
 *                    test the OpenCL path on a box that also has an
 *                    NVIDIA card).
 * tensor_to_cpu:     mirror image — copies back to a fresh host
 *                    buffer, frees device memory, sets device =
 *                    CF_DEVICE_CPU. Works for either backend.
 * cf_cuda_enabled:   1 if built with -DUSE_CUDA AND a CUDA device is
 *                    available at runtime, else 0.
 * cf_opencl_enabled: 1 if built with -DUSE_OPENCL AND an OpenCL
 *                    device is available at runtime, else 0.
 * Check these before calling tensor_to_gpu[_ex]() to fail gracefully
 * instead of hitting cforge_error().                               */
Tensor *tensor_to_gpu(Tensor *t);
Tensor *tensor_to_gpu_ex(Tensor *t, CF_GpuBackend backend);
Tensor *tensor_to_cpu(Tensor *t);
int     cf_cuda_enabled(void);
int     cf_opencl_enabled(void);

/* ── utilities ──────────────────────────────────────────────────── */
void tensor_print(const Tensor *t, const char *name);
int  tensor_shape_equal(const Tensor *a, const Tensor *b);

/* ── error handling (replaces raw assert) ───────────────────────── */
/* cforge_error: print message + file/line, then exit(1)            */
void cforge_error(const char *msg, const char *file, int line);

#define CF_CHECK(cond, msg) \
    do { if (!(cond)) cforge_error((msg), __FILE__, __LINE__); } while(0)

/* CF_CHECK_SHAPE: assert two tensors have matching sizes           */
#define CF_CHECK_SHAPE(a, b) \
    CF_CHECK((a)->size == (b)->size, \
             "Shape mismatch: tensors must have the same size")

/* CF_CHECK_2D: assert tensor is 2-dimensional                      */
#define CF_CHECK_2D(t) \
    CF_CHECK((t)->ndim == 2, "Expected a 2D tensor")

/* CF_CHECK_ALLOC: assert a malloc/create call succeeded            */
#define CF_CHECK_ALLOC(ptr) \
    CF_CHECK((ptr) != NULL, "Memory allocation failed")

#endif /* TENSOR_H */
