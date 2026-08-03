#ifndef NEURALC_MEMORY_H
#define NEURALC_MEMORY_H

#include <stddef.h>

/* ════════════════════════════════════════════════════════════════
 * memory.h — unified memory interface (static OR dynamic backend)
 * ════════════════════════════════════════════════════════════════
 * One switch, two backends, one API. Callers only ever see
 * neuralc_alloc()/neuralc_free(); which backend services the call is
 * decided entirely inside memory.c by USE_STATIC_MEMORY.
 *
 *   USE_STATIC_MEMORY 1  — STATIC backend (embedded systems)
 *     Every byte comes out of one fixed global array,
 *     `memory_pool[MAX_MEMORY_POOL]`, via a linear (bump) allocator:
 *     an `offset` that only ever moves forward. No malloc/free, no
 *     fragmentation handling, no free-list — by design. Overflow
 *     (request would exceed MAX_MEMORY_POOL) is checked on every
 *     call and refused by returning NULL, never by corrupting
 *     adjacent memory.
 *     neuralc_free() is a documented no-op in this mode: a linear
 *     allocator cannot reclaim one block out of the middle of the
 *     pool without fragmentation bookkeeping, which is explicitly
 *     out of scope here. The *only* way to reclaim static-mode
 *     memory is neuralc_memory_reset(), which rewinds `offset` to 0
 *     for the whole pool at once.
 *
 *   USE_STATIC_MEMORY 0  — DYNAMIC backend (desktop systems)
 *     Plain malloc()/free(), with a running byte counter so
 *     neuralc_memory_used() and MEMORY_DEBUG logging work uniformly
 *     across both backends. neuralc_free() here really does free the
 *     block and decrements the counter.
 *
 * ── enabling static mode ──────────────────────────────────────────
 *     #define USE_STATIC_MEMORY 1
 *     #define MAX_MEMORY_POOL   (1024 * 1024)   // bytes, tune per target
 *     #include "memory.h"
 *
 *   -- or on the compile line --
 *     gcc -DUSE_STATIC_MEMORY=1 -DMAX_MEMORY_POOL=1048576 ...
 *
 * Leaving USE_STATIC_MEMORY undefined (default 0) gives ordinary
 * malloc/free behavior with no fixed limit.
 */

#ifndef USE_STATIC_MEMORY
#define USE_STATIC_MEMORY 0         /* default: dynamic backend (malloc/free) */
#endif

/* Pool size in bytes, static backend only. Override before including
 * this header (or via -D) to fit the target's RAM budget. */
#ifndef MAX_MEMORY_POOL
#define MAX_MEMORY_POOL (1024 * 1024)   /* 1 MB default */
#endif

/* -DMEMORY_DEBUG=1 logs every allocation's size and the running
 * total usage after it. Off by default (0 cost when disabled). */
#ifndef MEMORY_DEBUG
#define MEMORY_DEBUG 0
#endif

/* ── public API ─────────────────────────────────────────────────── */

/* Must be called once before any neuralc_alloc() call.
 * Static backend : resets `offset` to 0 (pool starts empty).
 * Dynamic backend : resets the usage counter to 0.
 * Safe to call again later — behaves like a full reset either way. */
void neuralc_memory_init(void);

/* Reclaims everything allocated so far.
 * Static backend : rewinds `offset` to 0 — every pointer previously
 *                  returned by neuralc_alloc() must be treated as
 *                  invalid after this call (whole-pool reset, not
 *                  per-block — see header note above).
 * Dynamic backend : this call does NOT itself free any block (that
 *                  would double-free pointers callers may still be
 *                  holding); it is a bookkeeping reset only. Pair
 *                  every neuralc_alloc() with neuralc_free() in
 *                  dynamic builds as usual — see neuralc_memory_used()
 *                  to confirm nothing was missed before calling this. */
void neuralc_memory_reset(void);

/* Allocate `size` bytes.
 * Static backend : linear-allocated from memory_pool; returns NULL
 *                  if `size` would overflow MAX_MEMORY_POOL.
 * Dynamic backend : malloc(size); returns NULL on malloc failure,
 *                  same as malloc() itself.
 * `size == 0` returns NULL in both backends (no zero-length quirk to
 * reason about at call sites). */
void *neuralc_alloc(size_t size);

/* Release a block obtained from neuralc_alloc(). NULL is a no-op in
 * both backends.
 * Static backend : no-op beyond the NULL check — see header note.
 * Dynamic backend : real free(), decrements the usage counter. */
void neuralc_free(void *ptr);

/* Bytes currently considered "in use":
 * Static backend : the current `offset` (bytes handed out since the
 *                  last neuralc_memory_init()/neuralc_memory_reset()).
 * Dynamic backend : sum of sizes passed to neuralc_alloc() not yet
 *                  released via neuralc_free(). */
size_t neuralc_memory_used(void);

/* Aligned allocation API (e.g., 64-byte boundary for AVX2/AVX-512 SIMD) */
void *neuralc_aligned_alloc(size_t alignment, size_t size);
void neuralc_aligned_free(void *ptr);

#endif /* NEURALC_MEMORY_H */
