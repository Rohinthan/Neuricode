/*
 * cli.c — GPT-style streaming text generation CLI.
 *
 * DEVIATIONS FROM THE LITERAL SPEC (and why):
 *
 *   - `pipeline_load_model()` doesn't exist in this project's actual
 *     pipeline.h — the real loader is `pipeline_load()`, and it does
 *     NOT bundle a tokenizer (see tokenizer.h's separate
 *     `tokenizer_load()`). Used the real API throughout.
 *   - `tokenizer_decode(token)` returning a string doesn't match the
 *     real signature (`tokenizer_decode(t, tokens, count, buf, size)`,
 *     caller-allocated buffer). Used the real signature.
 *   - CLI usage: the spec's `./cli model.bin "prompt"` has no way to
 *     load a vocab file, but this tokenizer requires one. Usage here
 *     is `./cli model.bin vocab.txt ["prompt"]` — with the prompt
 *     arg: single-shot (generate once, exit). Without it: interactive
 *     REPL, matching how this CLI has worked throughout the project.
 *   - Repetition handling: replaced the earlier per-conversation
 *     "penalize only if the same token repeats >N times in a row"
 *     heuristic with the standard global `seen[]` scheme this spec
 *     asks for (penalize every token already emitted anywhere in the
 *     sequence, prompt included) — this is the same approach used by
 *     e.g. Hugging Face's `repetition_penalty` and is a strictly more
 *     standard design than the ad-hoc one before it.
 *   - Stop conditions: spec lists only EOS + max_tokens. Dropped the
 *     earlier newline-based stop to match. Easy to re-add if you want
 *     it back.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "pipeline.h"
#include "tokenizer.h"

/* ── sizing ─────────────────────────────────────────────────────── */
#define MAX_TOKENS      512    /* prompt + generated tokens, this call  */
#define MAX_NEW_TOKENS  50
#define INPUT_BYTES     4096
#define MAX_VOCAB       65536   /* static scratch-buffer cap             */
#define MAX_K           128     /* cap on top-k, independent of vocab    */

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

/* ── tunables ───────────────────────────────────────────────────── */
#ifndef TEMPERATURE
#ifdef NEURALC_TEMPERATURE
  #define TEMPERATURE NEURALC_TEMPERATURE
#else
  #define TEMPERATURE 0.8f
#endif
#endif
#define TOP_K           20
#define REPEAT_PENALTY  1.2f

static void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s <model.bin> <vocab.txt> [\"prompt\"]\n", program);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Section 9 — the five required building blocks
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * apply_temperature: logits[i] /= temp, in place. temp is clamped to
 * >= 0.05 — a near-zero temperature blows the scaled values up toward
 * +/-inf, which turns the softmax's exp() into NaN/inf downstream.
 */
static void apply_temperature(float *logits, int vocab, float temp) {
    if (temp < 0.05f) temp = 0.05f;
    for (int i = 0; i < vocab; i++)
        logits[i] /= temp;
}

/*
 * apply_repetition_penalty: for every token already seen (seen[i]
 * nonzero), push its logit DOWN — never up. This has to branch on
 * sign: dividing a positive logit by `penalty` shrinks it, but
 * dividing a *negative* logit by the same `penalty` makes it larger
 * (closer to zero, i.e. more likely) — a reward instead of a penalty.
 * Multiplying by `penalty` on the negative side keeps the direction
 * consistent regardless of the logit's sign.
 */
static void apply_repetition_penalty(float *logits, int vocab,
                                      const int *seen, float penalty) {
    for (int i = 0; i < vocab; i++) {
        if (!seen[i]) continue;
        if (logits[i] > 0.0f) logits[i] /= penalty;
        else                  logits[i] *= penalty;
    }
}

/*
 * top_k_filter: in place. Keeps the k highest logits untouched and
 * masks every other entry to -INFINITY, so a plain full-vocab softmax
 * afterward naturally assigns those entries ~0 probability.
 *
 * Uses partial selection (k passes of "find current max among
 * not-yet-picked") rather than a full sort — O(k*vocab), fine at this
 * project's fixed vocab_size=256 (see load_model.c's EXPECTED_VOCAB).
 * The "already picked" set is tracked via a static stamp counter
 * instead of a memset every call — see the comment on `stamp` below.
 */
static void top_k_filter(float *logits, int vocab, int k) {
    if (k <= 0) k = 1;
    if (k > vocab) k = vocab;
    if (k > MAX_K) k = MAX_K;

    static unsigned int used[MAX_VOCAB];
    static unsigned int stamp = 0;
    /* unsigned wraparound (UINT_MAX -> 0) is well-defined in C, unlike
     * the same wrap on a signed int, which is undefined behavior —
     * so a full reset on the rare wrap is correct, not just "usually
     * works". Happens once every ~4 billion calls. */
    if (++stamp == 0) {
        memset(used, 0, sizeof(used));
        stamp = 1;
    }

    int top_idx[MAX_K];
    int picked = 0;
    for (int pick = 0; pick < k; pick++) {
        int best = -1;
        for (int i = 0; i < vocab; i++) {
            if (used[i] == stamp) continue;
            float v = logits[i];
            if (v != v) continue;    /* NaN check w/o a libm call:
                                       * NaN is the only float never
                                       * equal to itself */
            if (best == -1 || v > logits[best]) best = i;
        }
        if (best == -1) break;       /* fewer than k finite candidates left */
        top_idx[picked++] = best;
        used[best] = stamp;
    }

    for (int i = 0; i < vocab; i++)
        if (used[i] != stamp) logits[i] = -INFINITY;

    (void)top_idx; /* indices themselves unneeded once the mask is applied */
}

/*
 * softmax: numerically stable, clamp-before-exp as specified.
 * Returns 0 on success (probs[] filled, sums to 1), -1 if the
 * distribution collapsed (every logit is -infinity/NaN, or the sum
 * came out non-finite/zero) — caller is expected to fall back to
 * argmax on the ORIGINAL logits in that case, not retry on probs[].
 */
static int softmax(const float *logits, float *probs, int vocab) {
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    if (!isfinite(max_logit))
        return -1;   /* every entry is -inf (fully masked) or vocab==0 */

    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        float diff = logits[i] - max_logit;
        if (!isfinite(diff)) diff = -INFINITY;   /* e.g. -inf - (-inf) => NaN */
        else if (diff < -20.0f) diff = -20.0f;   /* floor, not a NaN guard —
                                                    * prevents legitimate
                                                    * low-probability survivors
                                                    * from underflowing to a
                                                    * hard zero */
        float e = expf(diff);
        probs[i] = e;
        sum += e;
    }

    if (!(sum > 0.0) || !isfinite(sum))
        return -1;

    for (int i = 0; i < vocab; i++)
        probs[i] = (float)(probs[i] / sum);
    return 0;
}

/*
 * sample: draws one index from a normalized distribution via
 * cumulative-sum roulette. Falls back to argmax over probs[] only for
 * the floating-point-rounding edge case where the walk finishes
 * without cdf reaching r (extremely rare, not the "distribution
 * collapsed" case — that's handled by softmax()'s return value
 * before this is ever called).
 */
static int sample(const float *probs, int vocab) {
    float r = (float)rand() * (1.0f / ((float)RAND_MAX + 1.0f));
    float cdf = 0.0f;
    for (int i = 0; i < vocab; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    int best = 0;
    for (int i = 1; i < vocab; i++)
        if (probs[i] > probs[best]) best = i;
    return best;
}

/* argmax over raw, unmodified logits — the deterministic fallback
 * used when softmax() reports a collapsed distribution. Deliberately
 * takes the ORIGINAL logits (pre temperature/penalty/top-k), not the
 * masked working buffer, so a masking edge case can never itself be
 * the reason the fallback also fails. */
static int argmax(const float *logits, int vocab) {
    int best = 0;
    for (int i = 1; i < vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/*
 * generate_next_token: wires the five pieces above together for one
 * decoding step. No heap allocation — `work`/`probs` are static,
 * reused every call, sized once at MAX_VOCAB.
 */
static int generate_next_token(const float *raw_logits, int vocab,
                                const int *seen, float temperature,
                                int top_k, float penalty) {
    static float work[MAX_VOCAB];
    static float probs[MAX_VOCAB];

    /* temperature == 0 means "greedy" by definition (spec requirement,
     * not an approximation) — go straight to argmax on the raw,
     * unmodified logits and skip temperature/penalty/top-k/softmax/
     * sampling entirely. Reuses the exact same argmax() the collapse
     * fallback below uses, so both deterministic paths agree. A
     * negative temperature is nonsensical and treated the same way. */
    if (temperature <= 0.0f)
        return argmax(raw_logits, vocab);

    memcpy(work, raw_logits, (size_t)vocab * sizeof(float));

    apply_temperature(work, vocab, temperature);
    apply_repetition_penalty(work, vocab, seen, penalty);
    top_k_filter(work, vocab, top_k);

    if (softmax(work, probs, vocab) != 0)
        return argmax(raw_logits, vocab);   /* deterministic collapse fallback */

    return sample(probs, vocab);
}

/* ═══════════════════════════════════════════════════════════════════
 *  CLI plumbing
 * ═══════════════════════════════════════════════════════════════════ */

/* Runs one prompt to completion, streaming tokens to stdout as they're
 * generated. All buffers passed in are caller-owned and reused across
 * calls — no allocation happens in here. */
static void run_generation(Pipeline *pipeline, Tokenizer *tokenizer,
                            int vocab_size, int eos_id,
                            const char *prompt_text,
                            int *tokens, float *logits, int *seen) {
    int num_tokens = tokenizer_encode(tokenizer, prompt_text, tokens, MAX_TOKENS);
    if (num_tokens <= 0) {
        fprintf(stderr, "cli: input could not be tokenized\n");
        return;
    }

    /* Standard repetition-penalty scope: every token already in the
     * sequence counts as "seen", prompt included — not just tokens
     * generated so far. */
    memset(seen, 0, (size_t)vocab_size * sizeof(int));
    for (int i = 0; i < num_tokens; i++)
        seen[tokens[i]] = 1;

    pipeline_reset(pipeline);
    pipeline_forward(pipeline, tokens, num_tokens, logits);
    const float *cur_logits = logits;

    int max_new_tokens = MAX_NEW_TOKENS;
    if (max_new_tokens > MAX_TOKENS - num_tokens)
        max_new_tokens = MAX_TOKENS - num_tokens;

    for (int step = 0; step < max_new_tokens; step++) {
        int next_token = generate_next_token(cur_logits, vocab_size, seen,
                                              TEMPERATURE, TOP_K, REPEAT_PENALTY);
        if (next_token < 0 || next_token >= vocab_size) break;

        seen[next_token] = 1;
        tokens[num_tokens++] = next_token;

        /* STREAMING: decode and print only this one token. */
        char piece[256];
        int written = tokenizer_decode(tokenizer, &next_token, 1, piece, sizeof(piece));
        if (written > 0) {
            if (piece[0] == ' ')            /* manualy added for space */
                printf("%s", piece);        /* for front space and already in present in " == ' ' */
            else                            /* if else */
                printf(" %s", piece);       /* for back space for pieces*/

            fflush(stdout);
        }

        if (next_token == eos_id) break;   /* stop condition: EOS       */

        cur_logits = pipeline_step(pipeline, next_token);
        if (!cur_logits) break;
    }
    putchar('\n');
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    /* time(NULL) alone only has 1-second granularity; XOR in a stack
     * address (ASLR-randomized per process) so two runs launched in
     * the same second still diverge. */
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)&argc);

    Pipeline *pipeline = pipeline_load(argv[1]);
    if (!pipeline) {
        fprintf(stderr, "cli: failed to load model '%s'\n", argv[1]);
        return 1;
    }

    Tokenizer *tokenizer = tokenizer_load(argv[2]);
    if (!tokenizer) {
        fprintf(stderr, "cli: failed to load tokenizer '%s'\n", argv[2]);
        pipeline_free(pipeline);
        return 1;
    }

    int vocab_size = tokenizer_vocab_size(tokenizer);
    int eos_id = tokenizer_eos_id(tokenizer);
    if (vocab_size <= 0) {
        fprintf(stderr, "cli: tokenizer reports invalid vocab size (%d)\n", vocab_size);
        tokenizer_free(tokenizer);
        pipeline_free(pipeline);
        return 1;
    }
    if (vocab_size > MAX_VOCAB) {
        /* Fail fast rather than silently truncating the vocabulary —
         * a silent clamp would quietly drop real tokens from
         * consideration. This project's models are fixed at
         * vocab_size=256 (load_model.c's EXPECTED_VOCAB), so this
         * never actually triggers in practice. */
        fprintf(stderr, "cli: vocab_size %d exceeds MAX_VOCAB %d\n",
                vocab_size, MAX_VOCAB);
        tokenizer_free(tokenizer);
        pipeline_free(pipeline);
        return 1;
    }

    /* All generation-loop buffers allocated ONCE here, outside any
     * per-token path, and reused for every step / every prompt —
     * satisfies "no malloc/free inside the generation loop". */
    int   *tokens = calloc(MAX_TOKENS, sizeof(*tokens));
    float *logits = calloc((size_t)vocab_size, sizeof(*logits));
    int   *seen   = calloc((size_t)vocab_size, sizeof(*seen)); 
    if (!tokens || !logits || !seen) {
        fprintf(stderr, "cli: out of memory\n");
        free(tokens);
        free(logits);
        free(seen);
        tokenizer_free(tokenizer);
        pipeline_free(pipeline);
        return 1;
    }

    if (argc == 4) {
        /* single-shot mode: ./cli model.bin vocab.txt "prompt" */
        run_generation(pipeline, tokenizer, vocab_size, eos_id, argv[3],
                        tokens, logits, seen);
    } else {
        /* interactive REPL mode: ./cli model.bin vocab.txt */
        char input[INPUT_BYTES];
        while (1) {
            fputs("> ", stdout);
            fflush(stdout);
            if (!fgets(input, sizeof(input), stdin)) break;

            input[strcspn(input, "\n")] = '\0';
            if (input[0] == '\0') continue;

            run_generation(pipeline, tokenizer, vocab_size, eos_id, input,
                            tokens, logits, seen);
        }
    }

    free(tokens);
    free(logits);
    free(seen);
    tokenizer_free(tokenizer);
  //pipeline_free(pipeline);
    return 0;
}
