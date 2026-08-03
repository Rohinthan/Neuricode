/*
 * sanity_test.c — pipeline/tokenizer sanity check, run BEFORE any
 * sampling logic. Pure greedy (argmax) decoding, verbose diagnostics
 * at every stage, no sampling code at all.
 *
 * DEVIATIONS FROM THE LITERAL REQUEST (and why — same policy cli.c
 * follows: use the project's real APIs, not the spec's hypothetical
 * ones):
 *
 *   - `pipeline_load_model()` doesn't exist; the real loader is
 *     `pipeline_load()` (pipeline.h). Used that.
 *   - `pipeline_forward(pipeline, token)` returning `float *` doesn't
 *     match the real API. There are two real entry points instead:
 *       pipeline_forward(p, tokens[], seq_len, output)  — bulk-primes
 *         the model on a whole prompt, writes the LAST timestep's
 *         logits into a caller-allocated buffer, returns void.
 *       pipeline_step(p, token_id)                      — single-token
 *         step, ADVANCES hidden state, returns a pointer OWNED by the
 *         pipeline (must not be freed, valid only until the next
 *         call). This is what a generation loop should use per step.
 *     Phase 2's "first forward pass" uses pipeline_forward() to prime
 *     on the prompt; the Phase 4 generation loop uses pipeline_step().
 *   - `tokenizer_encode("hello")` returning a single `int` doesn't
 *     match the real signature — this tokenizer produces a sequence
 *     of ids, not one. Real signature: tokenizer_encode(t, text,
 *     tokens_out[], max_tokens) -> count written. Phase 3 below
 *     prints the whole id sequence, not a single token.
 *   - `tokenizer_decode(token)` returning `char *` doesn't match
 *     either — real signature takes an array of ids plus a
 *     caller-allocated output buffer: tokenizer_decode(t, tokens[],
 *     count, buf, buf_size) -> bytes written. Used that throughout.
 *
 * PHASE 6 (valgrind) is not code — run it externally once this links:
 *     valgrind --leak-check=full --show-leak-kinds=all \
 *         ./sanity_test model.bin vocab.txt "hello"
 *   Look for "invalid read/write" and "definitely lost" blocks. This
 *   file's own contribution to that check is discipline, not code: no
 *   malloc/free anywhere inside the generation loop (see PHASE 4/5).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pipeline.h"
#include "tokenizer.h"

#define MAX_TOKENS      256   /* prompt token cap, preallocated        */
#define MAX_VOCAB       65536 /* logits scratch cap, preallocated      */
#define GEN_STEPS       20    /* Phase 4/5: fixed greedy run length     */
#define LOGITS_PREVIEW  10    /* Phase 2: how many logits to print      */

/* ── Phase 2 helper: classify a logits buffer without allocating ── */
static void check_logits(const float *logits, int vocab_size, const char *label) {
    int has_nan = 0, has_inf = 0, all_zero = 1, all_same = 1;
    float first = logits[0];

    for (int i = 0; i < vocab_size; i++) {
        float v = logits[i];
        if (v != v) has_nan = 1;                 /* NaN != itself */
        if (isinf(v)) has_inf = 1;
        if (v != 0.0f) all_zero = 0;
        if (v != first) all_same = 0;
    }

    printf("[%s] first %d logits: ", label, LOGITS_PREVIEW);
    for (int i = 0; i < LOGITS_PREVIEW && i < vocab_size; i++)
        printf("%f ", logits[i]);
    printf("\n");

    printf("[%s] diagnostics: nan=%s inf=%s all_zero=%s all_identical=%s\n",
           label,
           has_nan   ? "YES (BAD — bad math in tensor ops)" : "no",
           has_inf   ? "YES (BAD — missing normalization)"  : "no",
           all_zero  ? "YES (BAD — weights not loaded)"     : "no",
           all_same  ? "YES (SUSPECT — check init/state)"   : "no");

    if (!has_nan && !has_inf && !all_zero && !all_same)
        printf("[%s] OK — looks like real logits.\n", label);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.bin> <vocab.txt> \"<prompt>\"\n", argv[0]);
        return 1;
    }
    const char *model_path  = argv[1];
    const char *vocab_path  = argv[2];
    const char *prompt_text = argv[3];

    /* ── PHASE 1: load ─────────────────────────────────────────── */
    printf("== PHASE 1: load ==\n");

    Pipeline *pipeline = pipeline_load(model_path);
    if (!pipeline) {
        fprintf(stderr, "FAIL: pipeline_load('%s') returned NULL\n", model_path);
        return 1;
    }
    printf("pipeline_load: OK\n");

    Tokenizer *tok = tokenizer_load(vocab_path);
    if (!tok) {
        fprintf(stderr, "FAIL: tokenizer_load('%s') returned NULL\n", vocab_path);
        pipeline_free(pipeline);
        return 1;
    }
    int vocab_size = tokenizer_vocab_size(tok);
    printf("tokenizer_load: OK, vocab_size=%d\n", vocab_size);
    if (vocab_size <= 0 || vocab_size > MAX_VOCAB) {
        fprintf(stderr, "FAIL: vocab_size %d out of expected range (1..%d)\n",
                vocab_size, MAX_VOCAB);
        tokenizer_free(tok);
        pipeline_free(pipeline);
        return 1;
    }

    /* All buffers used from here on are allocated ONCE, up front —
     * nothing inside PHASE 3/4/5 below calls malloc/free. */
    int   *prompt_tokens = malloc(MAX_TOKENS * sizeof(int));
    float *logits        = malloc((size_t)vocab_size * sizeof(float));
    char   decode_buf[1024];
    if (!prompt_tokens || !logits) {
        fprintf(stderr, "FAIL: out of memory\n");
        free(prompt_tokens); free(logits);
        tokenizer_free(tok); pipeline_free(pipeline);
        return 1;
    }

    /* ── PHASE 3: tokenizer check (encode/decode round trip) ──── */
    printf("\n== PHASE 3: tokenizer check ==\n");
    printf("Vocab size: %d\n", tokenizer_vocab_size(tok));
    printf("EOS id: %d\n", tokenizer_eos_id(tok));   /* a valid id or
                                                       * TOKENIZER_INVALID_ID
                                                       * (-1) are both fine —
                                                       * -1 just means this
                                                       * vocab declares no EOS */
    int n = tokenizer_encode(tok, prompt_text, prompt_tokens, MAX_TOKENS);
    if (n <= 0) {
        fprintf(stderr, "FAIL: tokenizer_encode('%s') produced %d tokens\n",
                prompt_text, n);
        free(prompt_tokens); free(logits);
        tokenizer_free(tok); pipeline_free(pipeline);
        return 1;
    }
    printf("Encoded \"%s\" -> %d tokens: ", prompt_text, n);
    for (int i = 0; i < n; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");

    decode_buf[0] = '\0';   /* defensive: guarantee a terminated buffer even
                              * if tokenizer_decode ever returns early on an
                              * unexpected multi-byte/UTF-8 token boundary */
    int written = tokenizer_decode(tok, prompt_tokens, n, decode_buf, sizeof(decode_buf));
    if (written < 0) {
        fprintf(stderr, "FAIL: tokenizer_decode returned -1\n");
    } else {
        /* Cosmetic only: a leading space here is the expected artifact of
         * add_dummy_prefix (see tokenizer.h), not an error — trimmed only
         * for this display line. Deliberately NOT applied to the per-token
         * decode calls in the Phase 4/5 loop below or in cli.c: there, that
         * same leading space is the actual inter-word separator in the
         * token stream, and trimming it would silently re-concatenate
         * generated words. */
        const char *display = decode_buf;
        if (display[0] == ' ') display++;
        printf("Decoded back -> \"%s\"\n", display);
        /* Compare against the same normalized (leading-space-trimmed)
         * string just displayed above, not raw decode_buf — otherwise
         * this verdict would report "differs" for every ordinary
         * prompt purely due to add_dummy_prefix's expected leading
         * marker, even though nothing is actually wrong. */
        printf("Round-trip %s\n",
               strcmp(display, prompt_text) == 0 ? "OK (normalized match)"
                                                  : "differs (check vocab/marker handling)");
    }

    /* ── PHASE 2: prime the model on the prompt, inspect logits ─ */
    printf("\n== PHASE 2: pipeline logits check ==\n");
    pipeline_reset(pipeline);                       /* start of sequence — the ONLY reset call */
    pipeline_forward(pipeline, prompt_tokens, n, logits);
    check_logits(logits, vocab_size, "prompt-primed");

    /* ── PHASE 4 + 5: greedy decode loop, state must persist ──── */
    printf("\n== PHASE 4/5: greedy decode (%d steps, no sampling) ==\n", GEN_STEPS);
    printf("(no pipeline_reset() call anywhere in this loop — that would\n"
           " wipe hidden state every step, the classic bug from the brief)\n");

    const float *cur_logits = logits;   /* from the priming forward() above */
    float prev_logits_sum = 0.0f;       /* to flag a stalled/non-updating state */
    int   has_prev = 0;                 /* explicit flag — no NaN-sentinel math */

    printf("Greedy output: ");
    for (int step = 0; step < GEN_STEPS; step++) {
        /* argmax over the CURRENT step's logits — no sampling */
        int next_token = 0;
        float max_val = cur_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (cur_logits[i] > max_val) {
                max_val = cur_logits[i];
                next_token = i;
            }
        }

        /* PHASE 5 sanity: sum of logits should change step to step if
         * hidden state is actually advancing. A frozen sum across
         * every step (beyond the first) means state isn't updating.
         * Relative tolerance (not a fixed absolute epsilon) so this
         * stays meaningful whether logits sit near 0 or run large. */
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) sum += cur_logits[i];
        if (has_prev) {
            float diff  = fabsf(sum - prev_logits_sum);
            float scale = fmaxf(1.0f, fabsf(prev_logits_sum));
            if (diff / scale < 1e-6f) {
                fprintf(stderr,
                    "\n[WARN] step %d: logits identical to previous step — "
                    "hidden state may not be advancing.\n", step);
            }
        }
        prev_logits_sum = sum;
        has_prev = 1;

        decode_buf[0] = '\0';
        written = tokenizer_decode(tok, &next_token, 1, decode_buf, sizeof(decode_buf));
        if (written > 0) {
            printf("%s", decode_buf);
            fflush(stdout);
        }

        int eos = tokenizer_eos_id(tok);
        if (eos != TOKENIZER_INVALID_ID && next_token == eos) {
            printf("\n[stopped: EOS at step %d]\n", step);
            break;
        }

        /* Advance state by exactly one token — this is the call that
         * must persist hidden state across iterations. */
        cur_logits = pipeline_step(pipeline, next_token);
        if (!cur_logits) {
            fprintf(stderr, "\n[FAIL] pipeline_step returned NULL at step %d\n", step);
            break;
        }
    }
    printf("\n");

    check_logits(cur_logits, vocab_size, "final-step");

    printf("\n== done ==\n"
           "Next: run under valgrind to check PHASE 6 —\n"
           "  valgrind --leak-check=full --show-leak-kinds=all %s %s %s \"%s\"\n",
           argv[0], model_path, vocab_path, prompt_text);

    free(prompt_tokens);
    free(logits);
    tokenizer_free(tok);
    pipeline_free(pipeline);
    return 0;
}
