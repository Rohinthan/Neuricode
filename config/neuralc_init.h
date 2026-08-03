/*
 * neuralc_init.c — Runtime initialization
 * Called automatically before main() via __attribute__((constructor)).
 * No user code changes needed.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "memory.h"

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "cuda_backend.h"
#endif

void neuralc_init(void) {

    /* ── bring up the memory subsystem first ──────────────────────
     * Must run before any neuralc_alloc() call. In static mode this
     * zeros `offset` (pool starts empty); in dynamic mode it resets
     * the usage counter. Declared in memory.h — without this include
     * the call below would be an implicit declaration (and, on
     * modern compilers defaulting to C99/C11+, a hard error rather
     * than just a warning). */
    neuralc_memory_init();

    /* ── detect real CPU core count (ChatGPT + Gemini) ── */
    long hw_cores = sysconf(_SC_NPROCESSORS_ONLN);
    /* Gemini: clamp — sysconf returns -1 on error */
    int max_cores = (hw_cores > 0) ? (int)hw_cores : 1;

#ifdef USE_OMP
    int target_threads = max_cores;  /* default: use all cores */
    int mode_auto      = 1;
    (void)mode_auto;  /* used in debug print only */

#ifdef NEURALC_HAS_CONFIG
    if (!NEURALC_USE_OMP) {
        /* OpenMP disabled in config — single threaded */
        target_threads = 1;
        mode_auto      = 0;
    } else if (!NEURALC_OMP_AUTO) {
        /* Manual mode — use configured thread count */
        target_threads = NEURALC_OMP_THREADS;
        mode_auto      = 0;
    }
    /* else: Auto mode — target_threads stays as max_cores */
#else
    /* No config file — check environment variable */
    const char *env = getenv("OMP_NUM_THREADS");
    if (env) {
        int parsed = atoi(env);
        if (parsed > 0) {
            target_threads = parsed;
            mode_auto      = 0;
        }
    }
#endif

    /*
     * core oversubscription guard
     * Never spawn more threads than physical cores available.
     * Spawning 16 threads on a 4-core CPU causes context switching
     * overhead that SLOWS DOWN training instead of speeding it up.
     */
    if (target_threads > max_cores) {
        target_threads = max_cores;
    }
    if (target_threads < 1) target_threads = 1;

    omp_set_num_threads(target_threads);

#ifdef NEURALC_HAS_CONFIG
    if (NEURALC_DEBUG) {
        printf("\n[neuralc] === Runtime Initialization ===\n");
        printf("[neuralc] CPU cores available : %d\n", max_cores);
        printf("[neuralc] OpenMP threads set  : %d (%s mode)\n",
               target_threads, mode_auto ? "AUTO" : "MANUAL");
        printf("[neuralc] =======================================\n\n");
    }
#endif /* NEURALC_HAS_CONFIG */

#endif /* USE_OMP */

    /* ── GPU (CUDA) status ─────────────────────────────────────────
     * Deliberately independent of the USE_OMP block above — GPU
     * status must be reported (and any config/build mismatch must be
     * warned about) even in builds without OpenMP.                 */
#ifdef NEURALC_HAS_CONFIG
#ifdef USE_CUDA
    int gpu_ready = cuda_available();
#else
    int gpu_ready = 0;
#endif

    if (NEURALC_USE_GPU && !gpu_ready) {
        fprintf(stderr,
            "[neuralc] WARNING: config requests GPU acceleration "
            "(NEURALC_USE_GPU=1), but %s\n"
            "[neuralc]          Falling back to CPU — call "
            "cf_cuda_enabled() before tensor_to_gpu() to avoid this.\n",
#ifdef USE_CUDA
            "no CUDA device was found at runtime."
#else
            "this binary was built without -DUSE_CUDA."
#endif
        );
    }
    if (NEURALC_DEBUG) {
        printf("[neuralc] GPU backend         : %s\n",
               gpu_ready ? "CUDA enabled, device available"
               : (
#ifdef USE_CUDA
                   "CUDA enabled, but no device detected"
#else
                   "CPU only (built without -DUSE_CUDA)"
#endif
               ));
    }
#endif /* NEURALC_HAS_CONFIG */
}

/*
 * __attribute__((constructor)) — GCC/Clang auto-call before main()
 * Same pattern used by PyTorch's ATen backend initialization.
 * note: this mimics torch.get_num_threads() hook behavior.
 */
#ifdef __GNUC__
__attribute__((constructor))
static void _neuralc_auto_init(void) {
    neuralc_init();
}
#endif
