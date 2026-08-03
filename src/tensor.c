#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
/* assert.h replaced by CF_CHECK */
#include <time.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "cuda_backend.h"
#endif

#ifdef USE_OPENCL
#include "opencl_backend.h"
#endif

/* ── helpers ─────────────────────────────────────────────────────── */

static size_t shape_size(const int *shape, int ndim) {
    size_t n = 1;
    for (int i = 0; i < ndim; i++) n *= (size_t)shape[i];
    return n;
}

/* Box-Muller: standard normal sample */
static float randn_sample(void) {
    static int   has_spare = 0;
    static float spare;
    if (has_spare) { has_spare = 0; return spare; }
    float u, v, s;
    do {
        u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        s = u*u + v*v;
    } while (s >= 1.0f || s == 0.0f);
    float mul = sqrtf(-2.0f * logf(s) / s);
    spare = v * mul;
    has_spare = 1;
    return u * mul;
}

/* ══════════════════════════════════════════════════════════════════
 * AUTOGRAD ENGINE — dynamic, tape-based reverse-mode differentiation
 * ══════════════════════════════════════════════════════════════════
 * Design summary:
 *   - Every tracked op pushes a GraphNode onto a single global tape
 *     (g_tape) and stamps `output->node = that node`.
 *   - A GraphNode's `parents` are the op's *input tensors*, not other
 *     GraphNodes directly — a parent is itself either another tracked
 *     op's output (parent->node != NULL, recursion continues) or a
 *     graph leaf (parent->node == NULL, recursion stops and the
 *     gradient is deposited straight into that leaf's ->grad).
 *   - Views (tensor_reshape()) are transparent to the graph: they
 *     carry no node of their own, but `effective_node()` below walks
 *     through view_base to find whatever node actually produced the
 *     underlying data, so gradient flow is never silently dropped
 *     just because a reshape sits in the middle of the graph.
 */

static GraphNode **g_tape      = NULL;
static int         g_tape_len  = 0;
static int         g_tape_cap  = 0;

/* Append a node to the global tape, growing the backing array as needed. */
static void tape_push(GraphNode *node) {
    if (g_tape_len == g_tape_cap) {
        g_tape_cap = g_tape_cap ? g_tape_cap * 2 : 64;
        g_tape = (GraphNode **)realloc(g_tape, (size_t)g_tape_cap * sizeof(GraphNode *));
        CF_CHECK_ALLOC(g_tape);
    }
    g_tape[g_tape_len++] = node;
}

/* Allocate + record a GraphNode for `output`, produced from `parents`
 * (n_parents inputs), with the given backward rule. Marks `output` as
 * part of the graph (requires_grad = 1, node = the new node). */
static GraphNode *tape_new_node(Tensor *output, Tensor **parents, int n_parents,
                                 void (*backward_fn)(GraphNode *node)) {
    GraphNode *node = (GraphNode *)malloc(sizeof(GraphNode));
    CF_CHECK_ALLOC(node);
    node->parents = (Tensor **)malloc((size_t)n_parents * sizeof(Tensor *));
    CF_CHECK_ALLOC(node->parents);
    memcpy(node->parents, parents, (size_t)n_parents * sizeof(Tensor *));
    node->n_parents   = n_parents;
    node->output      = output;
    node->backward_fn = backward_fn;
    node->visited     = 0;
    memset(node->ints, 0, sizeof(node->ints));

    output->node          = node;
    output->requires_grad = 1;
    tape_push(node);
    return node;
}

/* Resolves the tensor that actually owns the gradient buffer for `t`:
 * itself if it owns its data, otherwise its view_base (see the
 * Tensor.view_base doc in tensor.h). This is the single choke point
 * that gives views their "fallback safety" — every accumulation and
 * every lookup of a tensor's gradient goes through here. */
static Tensor *grad_home(Tensor *t) {
    return t->owns_data ? t : t->view_base;
}

/* Resolves the GraphNode that should be treated as having produced
 * `t`, following through a transparent view if `t` itself has no node
 * of its own. This lets backward() keep walking upstream even when a
 * reshape/flatten view sits between two tracked ops. */
static GraphNode *effective_node(Tensor *t) {
    if (!t) return NULL;
    if (t->node) return t->node;
    if (!t->owns_data && t->view_base) return t->view_base->node;
    return NULL;
}

/* Lazily allocate (zero-initialized) the gradient buffer backing `t`,
 * redirecting through grad_home() so a view never grows its own
 * orphaned buffer. */
static void ensure_grad_buffer(Tensor *t) {
    Tensor *home = grad_home(t);
    if (!home->grad) {
        home->grad = (float *)calloc(home->size, sizeof(float));
        CF_CHECK_ALLOC(home->grad);
    }
}

/* Accumulate (never overwrite) `n` gradient values into `t`'s grad
 * home. Accumulation (rather than assignment) is what makes fan-out
 * in the graph (a tensor used by more than one downstream op) sum
 * gradients correctly, matching the multivariable chain rule. */
static void accumulate_grad(Tensor *t, const float *delta, size_t n) {
    Tensor *home = grad_home(t);
    CF_CHECK(n == home->size,
             "autograd: gradient/element-count mismatch while accumulating "
             "into a tensor's grad buffer (shape changed under the graph?)");
    ensure_grad_buffer(t);
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(n > 4096)
#endif
    for (size_t i = 0; i < n; i++) home->grad[i] += delta[i];
}

/* Guard used by every tracked forward op before it records a node:
 * autograd math below runs plain CPU loops, so tracked tensors must
 * live on the CPU (mirrors the CF_IS_CUDA/CF_IS_OPENCL device-safety
 * checks used throughout the rest of this file). */
static void autograd_require_cpu(const Tensor *t, const char *op) {
    if (t->device != CF_DEVICE_CPU) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "autograd: '%s' gradient tracking is only supported for "
                 "CPU tensors — call tensor_to_cpu() before requires_grad", op);
        cforge_error(msg, __FILE__, __LINE__);
    }
}

/* Recursive DFS post-order: appends `node` only after all of its
 * (still-tracked) parents have been appended, giving a list where
 * dependencies precede dependents. tensor_backward() then walks this
 * list back-to-front, which is exactly reverse topological order —
 * the loss's node processed first, leaves' producing nodes last. */
static void topo_visit(GraphNode *node, GraphNode ***order, int *len, int *cap) {
    if (!node || node->visited == 2) return;
    CF_CHECK(node->visited != 1,
             "tensor_backward: cycle detected in the computation graph "
             "(a tensor must not depend on its own output)");
    node->visited = 1;
    for (int i = 0; i < node->n_parents; i++) {
        GraphNode *pn = effective_node(node->parents[i]);
        topo_visit(pn, order, len, cap);
    }
    node->visited = 2;

    if (*len == *cap) {
        *cap   = *cap ? *cap * 2 : 64;
        *order = (GraphNode **)realloc(*order, (size_t)(*cap) * sizeof(GraphNode *));
        CF_CHECK_ALLOC(*order);
    }
    (*order)[(*len)++] = node;
}

/* ── public autograd API ──────────────────────────────────────────── */

void tensor_requires_grad_(Tensor *t, int requires_grad) {
    if (requires_grad) autograd_require_cpu(t, "tensor_requires_grad_");
    t->requires_grad = requires_grad ? 1 : 0;
}

void tensor_zero_grad(Tensor *t) {
    Tensor *home = grad_home(t);
    if (home->grad) memset(home->grad, 0, home->size * sizeof(float));
}

void tensor_backward(Tensor *loss) {
    CF_CHECK(loss->size == 1,
              "tensor_backward: loss must be a scalar tensor (size == 1) — "
              "reduce it first (e.g. tensor_sum/tensor_mean)");
    CF_CHECK(loss->node != NULL,
              "tensor_backward: loss has no recorded graph history — was it "
              "produced by a tracked op (an input with requires_grad set)?");

    /* Reset visited markers before every pass so a second backward()
     * call over a still-live tape (e.g. after tensor_zero_grad on the
     * leaves) walks the graph again instead of seeing everything as
     * already-visited from a previous run. */
    for (int i = 0; i < g_tape_len; i++) g_tape[i]->visited = 0;

    /* Seed dLoss/dLoss = 1 directly into the loss tensor's own buffer. */
    ensure_grad_buffer(loss);
    grad_home(loss)->grad[0] += 1.0f;

    GraphNode **order = NULL;
    int len = 0, cap = 0;
    topo_visit(loss->node, &order, &len, &cap);

    /* Reverse-topological replay: dependents' backward_fn runs before
     * the parents whose gradient they feed. */
    for (int i = len - 1; i >= 0; i--) order[i]->backward_fn(order[i]);

    free(order);
}

void tensor_tape_clear(void) {
    for (int i = 0; i < g_tape_len; i++) {
        GraphNode *n = g_tape[i];
        if (n->output) n->output->node = NULL;  /* detach — no dangling ptr */
        free(n->parents);
        free(n);
    }
    free(g_tape);
    g_tape     = NULL;
    g_tape_len = 0;
    g_tape_cap = 0;
}

/* ── per-op backward rules ────────────────────────────────────────── */

/* out = a + b  =>  d/da = grad_out, d/db = grad_out */
static void add_backward_fn(GraphNode *node) {
    Tensor *out = node->output;
    if (!out->grad) return;                 /* no gradient ever reached here */
    accumulate_grad(node->parents[0], out->grad, out->size);
    accumulate_grad(node->parents[1], out->grad, out->size);
}

/* out = a - b  =>  d/da = grad_out, d/db = -grad_out */
static void sub_backward_fn(GraphNode *node) {
    Tensor *out = node->output;
    if (!out->grad) return;
    accumulate_grad(node->parents[0], out->grad, out->size);

    float *neg = (float *)malloc(out->size * sizeof(float));
    CF_CHECK_ALLOC(neg);
    for (size_t i = 0; i < out->size; i++) neg[i] = -out->grad[i];
    accumulate_grad(node->parents[1], neg, out->size);
    free(neg);
}

/* out = a * b (Hadamard)  =>  d/da = grad_out * b, d/db = grad_out * a */
static void mul_backward_fn(GraphNode *node) {
    Tensor *out = node->output;
    if (!out->grad) return;
    Tensor *a = node->parents[0], *b = node->parents[1];

    float *ga = (float *)malloc(out->size * sizeof(float));
    float *gb = (float *)malloc(out->size * sizeof(float));
    CF_CHECK_ALLOC(ga); CF_CHECK_ALLOC(gb);
    for (size_t i = 0; i < out->size; i++) {
        ga[i] = out->grad[i] * b->data[i];
        gb[i] = out->grad[i] * a->data[i];
    }
    accumulate_grad(a, ga, out->size);
    accumulate_grad(b, gb, out->size);
    free(ga); free(gb);
}

/* out[M,N] = a[M,K] @ b[K,N]
 *   d/da = grad_out @ b^T   ([M,N]@[N,K] -> [M,K])
 *   d/db = a^T @ grad_out   ([K,M]@[M,N] -> [K,N])
 * Computed with direct loops (rather than allocating transposed temps
 * + calling tensor_matmul) to keep this hot path allocation-light. */
static void matmul_backward_fn(GraphNode *node) {
    Tensor *out = node->output;
    if (!out->grad) return;
    Tensor *a = node->parents[0], *b = node->parents[1];
    int M = node->ints[0], K = node->ints[1], N = node->ints[2];

    float *ga = (float *)calloc((size_t)M * K, sizeof(float));
    float *gb = (float *)calloc((size_t)K * N, sizeof(float));
    CF_CHECK_ALLOC(ga); CF_CHECK_ALLOC(gb);

#ifdef USE_OMP
    #pragma omp parallel for collapse(2) schedule(static) if((size_t)M*K*N > 8192)
#endif
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float s = 0.0f;
            for (int j = 0; j < N; j++) s += out->grad[i*N + j] * b->data[k*N + j];
            ga[i*K + k] = s;
        }
    }
#ifdef USE_OMP
    #pragma omp parallel for collapse(2) schedule(static) if((size_t)M*K*N > 8192)
#endif
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int i = 0; i < M; i++) s += a->data[i*K + k] * out->grad[i*N + j];
            gb[k*N + j] = s;
        }
    }

    accumulate_grad(a, ga, (size_t)M * K);
    accumulate_grad(b, gb, (size_t)K * N);
    free(ga); free(gb);
}

/* out = permute(src, axis_order). Reuses tensor_permute_backward's
 * stride-mapping logic by wrapping the raw grad buffers in throwaway,
 * non-owning Tensor views (same shape metadata as out/src respectively,
 * data pointer swapped to the grad buffer) — no data is copied, and
 * the destination view targets grad_home(src) so the "view fallback
 * safety" (Task 1.4) applies here too if `src` is itself a reshape. */
static void permute_backward_fn(GraphNode *node) {
    Tensor *out = node->output;
    Tensor *src = node->parents[0];
    if (!out->grad) return;

    ensure_grad_buffer(src);
    Tensor *home = grad_home(src);

    Tensor grad_out_view = *out;          /* copy shape/ndim/etc.        */
    grad_out_view.data      = out->grad;  /* ...but read from the grad   */
    grad_out_view.owns_data = 0;

    Tensor grad_in_view = *src;           /* src's pre-permute shape     */
    grad_in_view.data      = home->grad;  /* ...write into home's grad   */
    grad_in_view.owns_data = 0;

    tensor_permute_backward(&grad_out_view, node->ints, &grad_in_view);
}

/* ── lifecycle ───────────────────────────────────────────────────── */

Tensor *tensor_create(const int *shape, int ndim) {
    CF_CHECK(ndim > 0 && ndim <= TENSOR_MAX_DIMS, "ndim out of range [1..TENSOR_MAX_DIMS]");
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->ndim = ndim;
    t->size = shape_size(shape, ndim);
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->data = (float *)malloc(t->size * sizeof(float));
    t->owns_data = 1;
    t->device = CF_DEVICE_CPU;
    t->gpu_backend = CF_GPU_NONE;
    /* autograd: fresh tensor starts as an untracked graph leaf */
    t->requires_grad = 0;
    t->grad = NULL;
    t->node = NULL;
    t->view_base = NULL;
    if (!t->data) { free(t); return NULL; }
    return t;
}

Tensor *tensor_zeros(const int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (t) memset(t->data, 0, t->size * sizeof(float));
    return t;
}

Tensor *tensor_ones(const int *shape, int ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++) t->data[i] = 1.0f;
    return t;
}

Tensor *tensor_rand(const int *shape, int ndim) {
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = (float)rand() / RAND_MAX;
    return t;
}

Tensor *tensor_randn(const int *shape, int ndim) {
    static int seeded = 0;
    if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = randn_sample();
    return t;
}

Tensor *tensor_clone(const Tensor *t) {
    /* NOTE (autograd): a clone is deliberately a fresh, untracked leaf —
     * tensor_create() already zero-initializes requires_grad/grad/node/
     * view_base for us. Cloning a tracked tensor detaches it from any
     * existing graph on purpose; if you need the clone's data to stay
     * differentiable, use an op (e.g. tensor_scale by 1.0) instead. */
    Tensor *c = tensor_create(t->shape, t->ndim);  /* always allocated on CPU */
    if (!c) return NULL;
#ifdef USE_CUDA
    if (t->device == CF_DEVICE_GPU && t->gpu_backend == CF_GPU_CUDA) {
        free(c->data);
        c->data = cuda_malloc_f32(t->size);
        c->device = CF_DEVICE_GPU;
        c->gpu_backend = CF_GPU_CUDA;
        cuda_memcpy_d2d(c->data, t->data, t->size);
        return c;
    }
#endif
#ifdef USE_OPENCL
    if (t->device == CF_DEVICE_GPU && t->gpu_backend == CF_GPU_OPENCL) {
        free(c->data);
        c->data = opencl_malloc_f32(t->size);
        c->device = CF_DEVICE_GPU;
        c->gpu_backend = CF_GPU_OPENCL;
        opencl_memcpy_d2d(c->data, t->data, t->size);
        return c;
    }
#endif
    memcpy(c->data, t->data, t->size * sizeof(float));
    return c;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    /* autograd: free this tensor's own gradient buffer, if any. Views
     * never allocate their own ->grad (see Tensor.view_base doc), so
     * this is always safe and never double-frees a base's buffer.
     * NOTE: we deliberately do NOT touch t->node here. GraphNodes are
     * owned by the global tape, not by individual tensors, because a
     * node's `parents` array may still be referenced by *other* nodes
     * further down the tape (they read parent->data / parent->node
     * during backward). Free the graph as a whole with
     * tensor_tape_clear() once backward() is done, rather than
     * piecemeal via tensor_free() on individual tensors — freeing a
     * tensor that's still a live dependency mid-graph is a use-after-
     * free waiting to happen, same as in any tape-based autograd. */
    free(t->grad);
    if (t->owns_data) {
#if defined(USE_CUDA) || defined(USE_OPENCL)
        if (t->device == CF_DEVICE_GPU) {
#ifdef USE_CUDA
            if (t->gpu_backend == CF_GPU_CUDA) { cuda_free_f32(t->data); free(t); return; }
#endif
#ifdef USE_OPENCL
            if (t->gpu_backend == CF_GPU_OPENCL) { opencl_free_f32(t->data); free(t); return; }
#endif
        }
#endif
        free(t->data);
    }
    free(t);
}

/* ── GPU residency ──────────────────────────────────────────────── */

int cf_cuda_enabled(void) {
#ifdef USE_CUDA
    return cuda_available();
#else
    return 0;
#endif
}

int cf_opencl_enabled(void) {
#ifdef USE_OPENCL
    return opencl_available();
#else
    return 0;
#endif
}

Tensor *tensor_to_gpu_ex(Tensor *t, CF_GpuBackend backend) {
    if (!t || t->device == CF_DEVICE_GPU) return t;
    CF_CHECK(t->owns_data, "tensor_to_gpu: cannot migrate a non-owning view");

    if (backend == CF_GPU_CUDA) {
#ifndef USE_CUDA
        cforge_error("tensor_to_gpu_ex: requested CUDA but library was built "
                     "without -DUSE_CUDA", __FILE__, __LINE__);
        return NULL;
#else
        float *dev = cuda_malloc_f32(t->size);
        cuda_memcpy_h2d(dev, t->data, t->size);
        free(t->data);
        t->data = dev;
        t->device = CF_DEVICE_GPU;
        t->gpu_backend = CF_GPU_CUDA;
        return t;
#endif
    } else if (backend == CF_GPU_OPENCL) {
#ifndef USE_OPENCL
        cforge_error("tensor_to_gpu_ex: requested OpenCL but library was built "
                     "without -DUSE_OPENCL", __FILE__, __LINE__);
        return NULL;
#else
        float *dev = opencl_malloc_f32(t->size);
        opencl_memcpy_h2d(dev, t->data, t->size);
        free(t->data);
        t->data = dev;
        t->device = CF_DEVICE_GPU;
        t->gpu_backend = CF_GPU_OPENCL;
        return t;
#endif
    }
    cforge_error("tensor_to_gpu_ex: no GPU backend requested", __FILE__, __LINE__);
    return NULL;
}

Tensor *tensor_to_gpu(Tensor *t) {
    if (!t || t->device == CF_DEVICE_GPU) return t;
#ifdef USE_CUDA
    if (cuda_available()) return tensor_to_gpu_ex(t, CF_GPU_CUDA);
#endif
#ifdef USE_OPENCL
    if (opencl_available()) return tensor_to_gpu_ex(t, CF_GPU_OPENCL);
#endif
    cforge_error("tensor_to_gpu: no usable GPU backend — library was built "
                 "without -DUSE_CUDA/-DUSE_OPENCL, or no device was found at "
                 "runtime (check cf_cuda_enabled()/cf_opencl_enabled() first)",
                 __FILE__, __LINE__);
    return NULL;
}

Tensor *tensor_to_cpu(Tensor *t) {
    if (!t || t->device == CF_DEVICE_CPU) return t;
    CF_CHECK(t->owns_data, "tensor_to_cpu: cannot migrate a non-owning view");
    float *host = (float *)malloc(t->size * sizeof(float));
    CF_CHECK_ALLOC(host);

#ifdef USE_CUDA
    if (t->gpu_backend == CF_GPU_CUDA) {
        cuda_memcpy_d2h(host, t->data, t->size);
        cuda_free_f32(t->data);
        t->data = host;
        t->device = CF_DEVICE_CPU;
        t->gpu_backend = CF_GPU_NONE;
        return t;
    }
#endif
#ifdef USE_OPENCL
    if (t->gpu_backend == CF_GPU_OPENCL) {
        opencl_memcpy_d2h(host, t->data, t->size);
        opencl_free_f32(t->data);
        t->data = host;
        t->device = CF_DEVICE_CPU;
        t->gpu_backend = CF_GPU_NONE;
        return t;
    }
#endif
    free(host);
    cforge_error("tensor_to_cpu: tensor is on GPU but no matching backend is "
                 "compiled in", __FILE__, __LINE__);
    return NULL;
}

/* ── element access ──────────────────────────────────────────────── */

float tensor_get(const Tensor *t, const int *idx) {
    size_t off = 0, stride = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        off    += idx[d] * stride;
        stride *= t->shape[d];
    }
    return t->data[off];
}

void tensor_set(Tensor *t, const int *idx, float val) {
    size_t off = 0, stride = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        off    += idx[d] * stride;
        stride *= t->shape[d];
    }
    t->data[off] = val;
}

/* ── element-wise ops ────────────────────────────────────────────── */

#ifdef USE_OMP
  #define OMP_ELWISE _Pragma("omp parallel for schedule(static)")
#else
  #define OMP_ELWISE
#endif

#if defined(USE_CUDA) || defined(USE_OPENCL)
  #define ELWISE2_GPU_HOOK(cuda_fn, opencl_fn, a, b, out)                 \
      if (CF_IS_CUDA(a)) {                                                \
          IF_CUDA(cuda_fn((a)->data, (b)->data, (out)->data, (a)->size); return;) \
      }                                                                    \
      if (CF_IS_OPENCL(a)) {                                              \
          IF_OPENCL(opencl_fn((a)->data, (b)->data, (out)->data, (a)->size); return;) \
      }
#else
  #define ELWISE2_GPU_HOOK(cuda_fn, opencl_fn, a, b, out)
#endif

#ifdef USE_CUDA
  #define IF_CUDA(x) x
#else
  #define IF_CUDA(x)
#endif
#ifdef USE_OPENCL
  #define IF_OPENCL(x) x
#else
  #define IF_OPENCL(x)
#endif

/* Shared registration helper for the binary elementwise ops below:
 * only tapes a node (and only pays for the malloc) when at least one
 * input actually requires_grad, so untracked/inference-only code
 * pays zero autograd overhead. */
static void record_elwise2(const Tensor *a, const Tensor *b, Tensor *out,
                            void (*backward_fn)(GraphNode *)) {
    if (!(a->requires_grad || b->requires_grad)) return;
    autograd_require_cpu(a, "elementwise op");
    autograd_require_cpu(b, "elementwise op");
    Tensor *parents[2] = { (Tensor *)a, (Tensor *)b };
    tape_new_node(out, parents, 2, backward_fn);
}

#define ELWISE2(name, cuda_fn, opencl_fn, expr, backward_fn)             \
void name(const Tensor *a, const Tensor *b, Tensor *out) {              \
    CF_CHECK_SHAPE(a, out); CF_CHECK_SHAPE(b, out);                      \
    ELWISE2_GPU_HOOK(cuda_fn, opencl_fn, a, b, out)                      \
    OMP_ELWISE                                                           \
    for (size_t i = 0; i < a->size; i++)                                 \
        out->data[i] = (expr);                                           \
    record_elwise2(a, b, out, backward_fn);                             \
}

ELWISE2(tensor_add, cuda_add, opencl_add, a->data[i] + b->data[i], add_backward_fn)
ELWISE2(tensor_sub, cuda_sub, opencl_sub, a->data[i] - b->data[i], sub_backward_fn)
ELWISE2(tensor_mul, cuda_mul, opencl_mul, a->data[i] * b->data[i], mul_backward_fn)

/* no GPU kernel for div on either backend (unused in any hot path) —
 * GPU tensors must be brought back with tensor_to_cpu() first */
void tensor_div(const Tensor *a, const Tensor *b, Tensor *out) {
    CF_CHECK_SHAPE(a, out); CF_CHECK_SHAPE(b, out);
#if defined(USE_CUDA) || defined(USE_OPENCL)
    CF_CHECK(a->device == CF_DEVICE_CPU,
             "tensor_div: no GPU kernel — call tensor_to_cpu() first");
#endif
    OMP_ELWISE
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = a->data[i] / b->data[i];
}

void tensor_scale(const Tensor *a, float s, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) { cuda_scale(a->data, s, out->data, a->size); return; }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) { opencl_scale(a->data, s, out->data, a->size); return; }
#endif
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = a->data[i] * s;
}
void tensor_add_scalar(const Tensor *a, float s, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = a->data[i] + s;
}
void tensor_neg(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = -a->data[i];
}
void tensor_abs(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = fabsf(a->data[i]);
}
void tensor_square(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = a->data[i] * a->data[i];
}
void tensor_sqrt_t(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = sqrtf(a->data[i]);
}
void tensor_exp(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = expf(a->data[i]);
}
void tensor_log_t(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) out->data[i] = logf(a->data[i]);
}
void tensor_clip(const Tensor *a, float lo, float hi, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    #ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
    #endif
    for (size_t i = 0; i < a->size; i++) {
        float v = a->data[i];
        out->data[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

/* ── reductions ──────────────────────────────────────────────────── */

float tensor_sum(const Tensor *t) {
    float s = 0.0f;
#ifdef USE_OMP
    #pragma omp parallel for reduction(+:s) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 0; i < t->size; i++) s += t->data[i];
    return s;
}
float tensor_mean(const Tensor *t) { return tensor_sum(t) / (float)t->size; }
float tensor_max(const Tensor *t) {
    float m = t->data[0];
#ifdef USE_OMP
    #pragma omp parallel for reduction(max:m) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 1; i < t->size; i++) if (t->data[i] > m) m = t->data[i];
    return m;
}
float tensor_min(const Tensor *t) {
    float m = t->data[0];
#ifdef USE_OMP
    #pragma omp parallel for reduction(min:m) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 1; i < t->size; i++) if (t->data[i] < m) m = t->data[i];
    return m;
}
float tensor_norm(const Tensor *t) {
    float s = 0.0f;
#ifdef USE_OMP
    #pragma omp parallel for reduction(+:s) schedule(static) \
        if(t->size > 4096)
#endif
    for (size_t i = 0; i < t->size; i++) s += t->data[i] * t->data[i];
    return sqrtf(s);
}

/* ── matrix ops ──────────────────────────────────────────────────── */

void tensor_matmul(const Tensor *a, const Tensor *b, Tensor *out) {
    /* a: [M,K]  b: [K,N]  out: [M,N] */
    CF_CHECK(a->ndim==2 && b->ndim==2 && out->ndim==2,
             "matmul: all tensors must be 2D");
    int M = a->shape[0], K = a->shape[1];
    int K2 = b->shape[0], N = b->shape[1];
    CF_CHECK(K == K2, "matmul: inner dimensions must match (a.cols == b.rows)");
    CF_CHECK(out->shape[0]==M && out->shape[1]==N,
             "matmul: output shape mismatch");

#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) {
        CF_CHECK(b->device == CF_DEVICE_GPU && out->device == CF_DEVICE_GPU,
                 "matmul: a, b, out must all be on the same device");
        cuda_matmul(a->data, b->data, out->data, M, K, N);
        return;
    }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) {
        CF_CHECK(b->device == CF_DEVICE_GPU && out->device == CF_DEVICE_GPU,
                 "matmul: a, b, out must all be on the same device");
        opencl_matmul(a->data, b->data, out->data, M, K, N);
        return;
    }
#endif

    // Zero output buffer
    memset(out->data, 0, (size_t)M * N * sizeof(float));

    // Workload threshold: 2 * M * K * N FLOPs.
    // Below 4 MFLOPs (e.g. 128x128x128 = 4.1 MFLOPs threshold), OpenMP thread dispatch
    // and task starvation outweigh parallel speedup.
    size_t total_flops = 2ULL * (size_t)M * (size_t)K * (size_t)N;
    int use_omp = 0;
#ifdef USE_OMP
    if (total_flops >= 4000000ULL && M >= omp_get_max_threads()) {
        use_omp = 1;
    }
#endif

    if (use_omp) {
        // Large matrix path: Tiled GEMM parallelized over outer row tiles with a single OpenMP region
        const int BLOCK_I = 64;
        const int BLOCK_J = 64;
        const int BLOCK_K = 64;

#ifdef USE_OMP
        #pragma omp parallel for schedule(static)
#endif
        for (int bi = 0; bi < M; bi += BLOCK_I) {
            int i_end = (bi + BLOCK_I < M) ? bi + BLOCK_I : M;
            for (int bj = 0; bj < N; bj += BLOCK_J) {
                int j_end = (bj + BLOCK_J < N) ? bj + BLOCK_J : N;
                for (int bk = 0; bk < K; bk += BLOCK_K) {
                    int k_end = (bk + BLOCK_K < K) ? bk + BLOCK_K : K;

                    for (int i = bi; i < i_end; i++) {
                        float *out_row = out->data + (size_t)i * N;
                        const float *a_row = a->data + (size_t)i * K;

                        for (int k = bk; k < k_end; k++) {
                            float a_ik = a_row[k];
                            const float *b_row = b->data + (size_t)k * N;

                            int j = bj;
#if defined(__AVX2__)
                            __m256 va = _mm256_set1_ps(a_ik);
                            for (; j <= j_end - 16; j += 16) {
                                __m256 vo0 = _mm256_loadu_ps(out_row + j);
                                __m256 vb0 = _mm256_loadu_ps(b_row + j);
                                vo0 = _mm256_fmadd_ps(va, vb0, vo0);
                                _mm256_storeu_ps(out_row + j, vo0);

                                __m256 vo1 = _mm256_loadu_ps(out_row + j + 8);
                                __m256 vb1 = _mm256_loadu_ps(b_row + j + 8);
                                vo1 = _mm256_fmadd_ps(va, vb1, vo1);
                                _mm256_storeu_ps(out_row + j + 8, vo1);
                            }
#endif
                            for (; j < j_end; j++) {
                                out_row[j] += a_ik * b_row[j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Small/Medium matrix path: Single-threaded SIMD contiguous row-major pass
        // Zero OpenMP barrier/fork latency, maximum L1 cache throughput
        for (int i = 0; i < M; i++) {
            float *out_row = out->data + (size_t)i * N;
            const float *a_row = a->data + (size_t)i * K;
            for (int k = 0; k < K; k++) {
                float a_ik = a_row[k];
                const float *b_row = b->data + (size_t)k * N;
                int j = 0;
#if defined(__AVX2__)
                __m256 va = _mm256_set1_ps(a_ik);
                for (; j <= N - 16; j += 16) {
                    __m256 vo0 = _mm256_loadu_ps(out_row + j);
                    __m256 vb0 = _mm256_loadu_ps(b_row + j);
                    vo0 = _mm256_fmadd_ps(va, vb0, vo0);
                    _mm256_storeu_ps(out_row + j, vo0);

                    __m256 vo1 = _mm256_loadu_ps(out_row + j + 8);
                    __m256 vb1 = _mm256_loadu_ps(b_row + j + 8);
                    vo1 = _mm256_fmadd_ps(va, vb1, vo1);
                    _mm256_storeu_ps(out_row + j + 8, vo1);
                }
#endif
                for (; j < N; j++) {
                    out_row[j] += a_ik * b_row[j];
                }
            }
        }
    }

    if (a->requires_grad || b->requires_grad) {
        autograd_require_cpu(a, "tensor_matmul");
        autograd_require_cpu(b, "tensor_matmul");
        Tensor *parents[2] = { (Tensor *)a, (Tensor *)b };
        GraphNode *node = tape_new_node(out, parents, 2, matmul_backward_fn);
        node->ints[0] = M; node->ints[1] = K; node->ints[2] = N;
    }
}

void tensor_transpose(const Tensor *a, Tensor *out) {
    CF_CHECK(a->ndim==2 && out->ndim==2, "transpose: tensors must be 2D");
    int R = a->shape[0], C = a->shape[1];
    CF_CHECK(out->shape[0]==C && out->shape[1]==R, "transpose: output shape must be [cols, rows]");
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            out->data[j*R + i] = a->data[i*C + j];
}

/* ── activations ─────────────────────────────────────────────────── */

void tensor_relu(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) { cuda_relu(a->data, out->data, a->size); return; }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) { opencl_relu(a->data, out->data, a->size); return; }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
}
void tensor_relu_grad(const Tensor *a, const Tensor *grad, Tensor *out) {
    CF_CHECK_SHAPE(a, grad); CF_CHECK_SHAPE(a, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) {
        cuda_relu_grad(a->data, grad->data, out->data, a->size); return;
    }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) {
        opencl_relu_grad(a->data, grad->data, out->data, a->size); return;
    }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = a->data[i] > 0.0f ? grad->data[i] : 0.0f;
}
void tensor_sigmoid(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) { cuda_sigmoid(a->data, out->data, a->size); return; }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) { opencl_sigmoid(a->data, out->data, a->size); return; }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
}
void tensor_sigmoid_grad(const Tensor *sig, const Tensor *grad, Tensor *out) {
    CF_CHECK_SHAPE(sig, grad); CF_CHECK_SHAPE(sig, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(sig)) {
        cuda_sigmoid_grad(sig->data, grad->data, out->data, sig->size); return;
    }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(sig)) {
        opencl_sigmoid_grad(sig->data, grad->data, out->data, sig->size); return;
    }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(sig->size > 4096)
#endif
    for (size_t i = 0; i < sig->size; i++) {
        float s = sig->data[i];
        out->data[i] = grad->data[i] * s * (1.0f - s);
    }
}
void tensor_tanh_t(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) { cuda_tanh_f(a->data, out->data, a->size); return; }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) { opencl_tanh_f(a->data, out->data, a->size); return; }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(a->size > 4096)
#endif
    for (size_t i = 0; i < a->size; i++)
        out->data[i] = tanhf(a->data[i]);
}
void tensor_tanh_grad(const Tensor *th, const Tensor *grad, Tensor *out) {
    CF_CHECK_SHAPE(th, grad); CF_CHECK_SHAPE(th, out);
#ifdef USE_CUDA
    if (CF_IS_CUDA(th)) {
        cuda_tanh_grad(th->data, grad->data, out->data, th->size); return;
    }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(th)) {
        opencl_tanh_grad(th->data, grad->data, out->data, th->size); return;
    }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(th->size > 4096)
#endif
    for (size_t i = 0; i < th->size; i++) {
        float t = th->data[i];
        out->data[i] = grad->data[i] * (1.0f - t*t);
    }
}

/* row-wise softmax for 2-D [batch, classes] */
void tensor_softmax(const Tensor *a, Tensor *out) {
    CF_CHECK_SHAPE(a, out);
    int rows = (a->ndim >= 2) ? a->shape[0] : 1;
    int cols = (int)(a->size / rows);
#ifdef USE_CUDA
    if (CF_IS_CUDA(a)) { cuda_softmax_rows(a->data, out->data, rows, cols); return; }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(a)) { opencl_softmax_rows(a->data, out->data, rows, cols); return; }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(rows > 16)
#endif
    for (int r = 0; r < rows; r++) {
        float *src = a->data   + r*cols;
        float *dst = out->data + r*cols;
        float mx = src[0];
        for (int c = 1; c < cols; c++) if (src[c] > mx) mx = src[c];
        float s = 0.0f;
        for (int c = 0; c < cols; c++) { dst[c] = expf(src[c]-mx); s += dst[c]; }
        for (int c = 0; c < cols; c++) dst[c] /= s;
    }
}

/* ── shape ops ───────────────────────────────────────────────────── */

Tensor *tensor_reshape(Tensor *t, const int *new_shape, int new_ndim) {
    size_t new_size = shape_size(new_shape, new_ndim);
    CF_CHECK(new_size == t->size, "reshape: total elements must stay the same");
    Tensor *r = (Tensor *)malloc(sizeof(Tensor));
    if (!r) return NULL;
    r->data      = t->data;
    r->size      = t->size;
    r->ndim      = new_ndim;
    r->owns_data = 0;   /* view: does not own the data */
    r->device    = t->device;
    r->gpu_backend = t->gpu_backend;
    memcpy(r->shape, new_shape, new_ndim * sizeof(int));

    /* autograd: this view never gets its own ->grad buffer — gradients
     * aimed at it are redirected to whichever tensor actually owns the
     * allocation (chain through if `t` is itself already a view, e.g.
     * reshape-of-a-reshape, so every view in the chain resolves to the
     * one true base in a single hop). requires_grad propagates through
     * automatically since reshape/flatten is just a shape reinterpret.
     * `node` stays NULL: reshape itself is not taped as an op — the
     * topological walk resolves the *effective* producing node for a
     * view straight through to view_base->node (see effective_node()
     * below), so upstream history through the view is never lost. */
    r->requires_grad = t->requires_grad;
    r->grad          = NULL;
    r->node          = NULL;
    r->view_base     = t->owns_data ? t : t->view_base;
    return r;
}

void tensor_fill(Tensor *t, float val) {
#ifdef USE_CUDA
    if (CF_IS_CUDA(t)) { cuda_fill(t->data, val, t->size); return; }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(t)) { opencl_fill(t->data, val, t->size); return; }
#endif
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(t->size > 4096)
#endif
    for (size_t i = 0; i < t->size; i++) t->data[i] = val;
}

void tensor_copy_data(Tensor *dst, const Tensor *src) {
    CF_CHECK(dst->size == src->size, "copy_data: size mismatch");
#ifdef USE_CUDA
    if (CF_IS_CUDA(src) || CF_IS_CUDA(dst)) {
        CF_CHECK(src->device == dst->device && src->gpu_backend == dst->gpu_backend,
                 "copy_data: src/dst on different devices — use tensor_to_cpu/gpu first");
        cuda_memcpy_d2d(dst->data, src->data, src->size);
        return;
    }
#endif
#ifdef USE_OPENCL
    if (CF_IS_OPENCL(src) || CF_IS_OPENCL(dst)) {
        CF_CHECK(src->device == dst->device && src->gpu_backend == dst->gpu_backend,
                 "copy_data: src/dst on different devices — use tensor_to_cpu/gpu first");
        opencl_memcpy_d2d(dst->data, src->data, src->size);
        return;
    }
#endif
    memcpy(dst->data, src->data, src->size * sizeof(float));
}

/* ── layout transforms (axis permutation) ─────────────────────────── */

/* Validates axis_order is a genuine permutation of [0..ndim-1] and
 * matches src's rank — shared by forward and backward so a malformed
 * axis_order is caught the same way (and at the same call site) on
 * either pass. */
static void validate_axis_order(const int *axis_order, int ndim) {
    CF_CHECK(ndim > 0 && ndim <= TENSOR_MAX_DIMS,
             "permute: ndim out of range [1..TENSOR_MAX_DIMS]");
    int seen[TENSOR_MAX_DIMS] = {0};
    for (int d = 0; d < ndim; d++) {
        int a = axis_order[d];
        CF_CHECK(a >= 0 && a < ndim, "permute: axis_order entry out of bounds");
        CF_CHECK(!seen[a], "permute: axis_order must be a permutation (no repeated axis)");
        seen[a] = 1;
    }
}

/* Row-major strides for a given shape (stride[ndim-1] = 1, growing
 * leftwards) — the same displacement math used by tensor_get/tensor_set,
 * factored out here since both permute and its backward need it twice
 * (once for the "from" layout, once for the "to" layout). */
static void row_major_strides(const int *shape, int ndim, size_t *strides) {
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * (size_t)shape[d + 1];
}

Tensor *tensor_permute(Tensor *src, const int *axis_order, int ndim) {
    CF_CHECK(ndim == src->ndim, "permute: axis_order length must match src->ndim");
    validate_axis_order(axis_order, ndim);
    CF_CHECK(src->device == CF_DEVICE_CPU,
             "permute: only supported for CPU tensors — call tensor_to_cpu() first");

    int new_shape[TENSOR_MAX_DIMS];
    for (int d = 0; d < ndim; d++) new_shape[d] = src->shape[axis_order[d]];

    Tensor *out = tensor_create(new_shape, ndim);
    CF_CHECK_ALLOC(out);

    size_t src_strides[TENSOR_MAX_DIMS], out_strides[TENSOR_MAX_DIMS];
    row_major_strides(src->shape, ndim, src_strides);
    row_major_strides(new_shape,  ndim, out_strides);

    /* Multi-dimensional stride mapping: for every flat index in the
     * *output*, decompose it into per-axis coordinates in the new
     * layout, then re-project those coordinates onto the *source*'s
     * strides through axis_order to find the single source element
     * that belongs there. This is the O(size * ndim) core of the op —
     * no intermediate index buffers, no extra passes. */
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(out->size > 4096)
#endif
    for (size_t lin = 0; lin < out->size; lin++) {
        size_t rem = lin;
        int out_idx[TENSOR_MAX_DIMS];
        for (int d = 0; d < ndim; d++) {
            out_idx[d] = (int)(rem / out_strides[d]);
            rem       %= out_strides[d];
        }
        size_t src_off = 0;
        for (int d = 0; d < ndim; d++)
            src_off += (size_t)out_idx[d] * src_strides[axis_order[d]];
        out->data[lin] = src->data[src_off];
    }

    /* autograd: tape this permute if its input is tracked. */
    if (src->requires_grad) {
        autograd_require_cpu(src, "tensor_permute");
        GraphNode *node = tape_new_node(out, &src, 1, permute_backward_fn);
        for (int d = 0; d < ndim; d++) node->ints[d] = axis_order[d];
    }
    return out;
}

void tensor_permute_backward(const Tensor *grad_out, const int *axis_order, Tensor *grad_in) {
    int ndim = grad_in->ndim;
    CF_CHECK(grad_out->ndim == ndim, "permute_backward: rank mismatch between grad_out and grad_in");
    CF_CHECK(grad_out->size == grad_in->size, "permute_backward: element-count mismatch");
    validate_axis_order(axis_order, ndim);

    size_t out_strides[TENSOR_MAX_DIMS], in_strides[TENSOR_MAX_DIMS];
    row_major_strides(grad_out->shape, ndim, out_strides);
    row_major_strides(grad_in->shape,  ndim, in_strides);

    /* Exact mirror of tensor_permute's forward stride mapping, with
     * read/write roles reversed: walk grad_out linearly, project each
     * coordinate back onto grad_in's original (pre-permute) strides
     * through axis_order, and scatter-add (never overwrite — this
     * must compose safely with any other gradient already sitting in
     * grad_in, e.g. when src feeds more than one downstream op).
     * validate_axis_order() above guarantees axis_order is a true
     * permutation, so lin -> in_off is a bijection: every grad_in
     * element is touched by exactly one `lin`, making the += safe to
     * parallelize (no two threads ever write the same offset). */
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(grad_out->size > 4096)
#endif
    for (size_t lin = 0; lin < grad_out->size; lin++) {
        size_t rem = lin;
        int out_idx[TENSOR_MAX_DIMS];
        for (int d = 0; d < ndim; d++) {
            out_idx[d] = (int)(rem / out_strides[d]);
            rem       %= out_strides[d];
        }
        size_t in_off = 0;
        for (int d = 0; d < ndim; d++)
            in_off += (size_t)out_idx[d] * in_strides[axis_order[d]];
        grad_in->data[in_off] += grad_out->data[lin];
    }
}

/* ── utilities ───────────────────────────────────────────────────── */

void tensor_print(const Tensor *t, const char *name) {
    printf("Tensor '%s'  shape=[", name ? name : "?");
    for (int d = 0; d < t->ndim; d++) {
        printf("%d", t->shape[d]);
        if (d < t->ndim-1) printf(",");
    }
    printf("]  size=%zu\n", t->size);
    if (t->ndim == 1) {
        for (size_t i = 0; i < t->size; i++) printf("  %.4f", t->data[i]);
        printf("\n");
    } else if (t->ndim == 2) {
        int R = t->shape[0], C = t->shape[1];
        for (int i = 0; i < R; i++) {
            printf("  [");
            for (int j = 0; j < C; j++) printf(" %8.4f", t->data[i*C+j]);
            printf(" ]\n");
        }
    } else {
        /* flat dump for higher dims */
        for (size_t i = 0; i < t->size; i++) printf("  %.4f", t->data[i]);
        printf("\n");
    }
}

int tensor_shape_equal(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (int d = 0; d < a->ndim; d++)
        if (a->shape[d] != b->shape[d]) return 0;
    return 1;
}

/* ── argmax / argmin ─────────────────────────────────────────────── */

int tensor_argmax(const Tensor *t) {
    int best = 0;
    float best_val = t->data[0];
    for (size_t i = 1; i < t->size; i++)
        if (t->data[i] > best_val) { best_val = t->data[i]; best = (int)i; }
    return best;
}

int tensor_argmin(const Tensor *t) {
    int best = 0;
    float best_val = t->data[0];
    for (size_t i = 1; i < t->size; i++)
        if (t->data[i] < best_val) { best_val = t->data[i]; best = (int)i; }
    return best;
}

/* per-row argmax for [batch, classes] — fills caller-supplied int array */
void tensor_argmax_rows(const Tensor *t, int *out) {
    CF_CHECK_2D(t);
    int rows = t->shape[0], cols = t->shape[1];
#ifdef USE_OMP
    #pragma omp parallel for schedule(static) if(rows > 16)
#endif
    for (int r = 0; r < rows; r++) {
        int   best     = 0;
        float best_val = t->data[r * cols];
        for (int c = 1; c < cols; c++) {
            float v = t->data[r * cols + c];
            if (v > best_val) { best_val = v; best = c; }
        }
        out[r] = best;
    }
}

/* ── axis reductions ─────────────────────────────────────────────── */

void tensor_sum_axis(const Tensor *t, int axis, Tensor *out) {
    CF_CHECK_2D(t);
    int rows = t->shape[0], cols = t->shape[1];
    tensor_fill(out, 0.0f);

    if (axis == 0) {
        /*
         * Sum across rows → out shape [cols]
         *
         * GEMINI FIX: old loop (r outer, c inner) caused data race —
         * multiple threads writing to same out->data[c].
         *
         * FIX: flip to c outer, r inner. Each thread owns one column
         * exclusively — no race, no atomic needed.
         */
        CF_CHECK(out->size == (size_t)cols,
                 "sum_axis(0): out must have size == cols");
#ifdef USE_OMP
        #pragma omp parallel for schedule(static) if(cols > 64)
#endif
        for (int c = 0; c < cols; c++) {
            float sum = 0.0f;
            for (int r = 0; r < rows; r++)
                sum += t->data[r * cols + c];
            out->data[c] = sum;   /* single write per thread — safe */
        }
    } else {
        /*
         * Sum across cols → out shape [rows]
         * Each row is independent — parallelizing over r is safe.
         */
        CF_CHECK(out->size == (size_t)rows,
                 "sum_axis(1): out must have size == rows");
#ifdef USE_OMP
        #pragma omp parallel for schedule(static) if(rows > 16)
#endif
        for (int r = 0; r < rows; r++) {
            float sum = 0.0f;
            for (int c = 0; c < cols; c++)
                sum += t->data[r * cols + c];
            out->data[r] = sum;   /* single write per thread — safe */
        }
    }
}

void tensor_mean_axis(const Tensor *t, int axis, Tensor *out) {
    tensor_sum_axis(t, axis, out);
    float div = (axis == 0) ? (float)t->shape[0] : (float)t->shape[1];
    tensor_scale(out, 1.0f / div, out);  /* tensor_scale already parallel */
}

/* ── error handler ───────────────────────────────────────────────── */

void cforge_error(const char *msg, const char *file, int line) {
    fprintf(stderr, "\n[cforge ERROR] %s\n  at %s:%d\n", msg, file, line);
    exit(1);
}
