#define _POSIX_C_SOURCE 200809L /* for rand_r() */

/*
 * sampler.c — sampling strategies, split out of the old model.c so
 * new strategies (top_k/top_p here; anything else later) don't touch
 * the inference loop at all.
 *
 * `probs` passed into sampler_pick() is assumed to already be a
 * softmax distribution (PROBABILITIES, not logits) — e.g. straight
 * out of a Dense layer with ACT_SOFTMAX. Temperature is therefore
 * applied by converting back to log-space first (logf(p)), THEN
 * scaling by 1/temperature, THEN re-softmaxing — this is what
 * PyTorch/TF-style temperature sampling actually does, and it
 * recovers the original distribution exactly at temperature == 1.
 * Naively dividing a probability by temperature and re-normalizing
 * (skipping the log step) distorts the distribution incorrectly.
 */

#include "sampler.h"
#include <math.h>
#include <stdlib.h>

/* ── constructors ──────────────────────────────────────────────── */

Sampler sampler_greedy(void) {
    Sampler s;
    s.kind               = SAMPLE_GREEDY;
    s.temperature        = 0.0f;
    s.top_k              = 0;
    s.top_p              = 0.0f;
    s.repetition_penalty = 1.0f;
    s.frequency_penalty  = 0.0f;
    s.rng_state          = 0;
    return s;
}

Sampler sampler_temperature(float temperature, unsigned seed) {
    Sampler s;
    s.kind               = SAMPLE_TEMPERATURE;
    s.temperature        = temperature;
    s.top_k              = 0;
    s.top_p              = 0.0f;
    s.repetition_penalty = 1.0f;
    s.frequency_penalty  = 0.0f;
    s.rng_state          = seed;
    return s;
}

Sampler sampler_top_k(int top_k, float temperature, unsigned seed) {
    Sampler s;
    s.kind               = SAMPLE_TOP_K;
    s.temperature        = temperature;
    s.top_k              = top_k;
    s.top_p              = 0.0f;
    s.repetition_penalty = 1.0f;
    s.frequency_penalty  = 0.0f;
    s.rng_state          = seed;
    return s;
}

Sampler sampler_top_p(float top_p, float temperature, unsigned seed) {
    Sampler s;
    s.kind               = SAMPLE_TOP_P;
    s.temperature        = temperature;
    s.top_k              = 0;
    s.top_p              = top_p;
    s.repetition_penalty = 1.0f;
    s.frequency_penalty  = 0.0f;
    s.rng_state          = seed;
    return s;
}

void sampler_apply_penalties(float *logits, int vocab_size,
                             const int *tokens, int num_tokens,
                             float repetition_penalty, float frequency_penalty) {
    if (!logits || vocab_size <= 0 || !tokens || num_tokens <= 0) return;
    if (repetition_penalty == 1.0f && frequency_penalty == 0.0f) return;

    for (int i = 0; i < num_tokens; i++) {
        int token = tokens[i];
        if (token >= 0 && token < vocab_size) {
            int count = 0;
            for (int j = 0; j < num_tokens; j++) {
                if (tokens[j] == token) count++;
            }

            if (repetition_penalty != 1.0f) {
                if (logits[token] < 0.0f) {
                    logits[token] *= repetition_penalty;
                } else {
                    logits[token] /= repetition_penalty;
                }
            }

            if (frequency_penalty != 0.0f) {
                logits[token] -= frequency_penalty * (float)count;
            }
        }
    }
}

/* ── internal helpers ──────────────────────────────────────────── */

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* logf(p)/temperature -> softmax, written into probs_scratch.
 * logits_scratch is used as working space. Returns nothing; caller
 * reads probs_scratch afterward. Both buffers length n. */
static void temperature_softmax(const float *probs, float temperature, int n,
                                 float *logits_scratch, float *probs_scratch) {
    for (int i = 0; i < n; i++)
        logits_scratch[i] = logf(probs[i] + 1e-9f) / temperature;

    float max_logit = logits_scratch[0];
    for (int i = 1; i < n; i++)
        if (logits_scratch[i] > max_logit) max_logit = logits_scratch[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = exp((double)(logits_scratch[i] - max_logit));
        probs_scratch[i] = (float)e;
        sum += e;
    }
    if (sum <= 0.0) sum = 1.0;
    for (int i = 0; i < n; i++)
        probs_scratch[i] = (float)(probs_scratch[i] / sum);
}

/* Multinomial draw from a normalized distribution, using this
 * sampler's own rand_r stream (not the global rand()). */
static int multinomial(Sampler *s, const float *probs, int n) {
    double r   = (double)rand_r(&s->rng_state) / ((double)RAND_MAX + 1.0);
    double cum = 0.0;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r <= cum) return i;
    }
    return n - 1; /* floating-point rounding fallback */
}

/* ── entry point ───────────────────────────────────────────────── */

int sampler_pick(Sampler *s, const float *probs, int vocab_size,
                  float *logits_scratch, float *probs_scratch) {
    if (!s || !probs || vocab_size <= 0 || !logits_scratch || !probs_scratch)
        return -1;

    if (s->kind == SAMPLE_GREEDY)
        return argmax(probs, vocab_size);

    if (s->temperature <= 0.0f)
        return -1; /* GREEDY is the only kind allowed to skip temperature;
                     * every other kind must supply one > 0 */

    switch (s->kind) {
    case SAMPLE_TEMPERATURE: {
        temperature_softmax(probs, s->temperature, vocab_size,
                             logits_scratch, probs_scratch);
        return multinomial(s, probs_scratch, vocab_size);
    }

    case SAMPLE_TOP_K: {
        if (s->top_k <= 0) return -1;
        int k = s->top_k < vocab_size ? s->top_k : vocab_size;

        temperature_softmax(probs, s->temperature, vocab_size,
                             logits_scratch, probs_scratch);

        /* Partial selection sort: pull the top-k indices to the front
         * of a small index scratch (reusing logits_scratch as an int
         * bitmap-by-value would clobber it, so track indices inline
         * via k passes of "find current max, zero it out, remember
         * where"). O(k * vocab_size) — fine for k in the tens; for a
         * very large k on a very large vocab, a real partial-sort
         * (e.g. introselect) would be worth it, per the note in the
         * original review. */
        double kept_sum = 0.0;
        for (int pick = 0; pick < k; pick++) {
            int best = -1;
            for (int i = 0; i < vocab_size; i++) {
                if (probs_scratch[i] < 0.0f) continue; /* already excluded/kept */
                if (best == -1 || probs_scratch[i] > probs_scratch[best])
                    best = i;
            }
            kept_sum += probs_scratch[best];
            /* mark as "kept" by negating (restored below); values are
             * probabilities so always >= 0 before this pass touches them */
            probs_scratch[best] = -probs_scratch[best] - 1e-12f;
        }
        /* zero out everything not kept, restore + renormalize the rest */
        if (kept_sum <= 0.0) kept_sum = 1.0;
        for (int i = 0; i < vocab_size; i++) {
            if (probs_scratch[i] < 0.0f)
                probs_scratch[i] = (float)((-probs_scratch[i] - 1e-12f) / kept_sum);
            else
                probs_scratch[i] = 0.0f;
        }
        return multinomial(s, probs_scratch, vocab_size);
    }

    case SAMPLE_TOP_P: {
        if (s->top_p <= 0.0f || s->top_p > 1.0f) return -1;

        temperature_softmax(probs, s->temperature, vocab_size,
                             logits_scratch, probs_scratch);

        /* Greedily accumulate the largest probabilities until the
         * running sum reaches top_p, same repeated-argmax approach as
         * TOP_K above (no full sort needed — nucleus size is usually
         * small relative to vocab_size). */
        double cum = 0.0;
        int remaining = vocab_size;
        while (cum < (double)s->top_p && remaining > 0) {
            int best = -1;
            for (int i = 0; i < vocab_size; i++) {
                if (probs_scratch[i] < 0.0f) continue;
                if (best == -1 || probs_scratch[i] > probs_scratch[best])
                    best = i;
            }
            cum += probs_scratch[best];
            probs_scratch[best] = -probs_scratch[best] - 1e-12f;
            remaining--;
        }
        double kept_sum = cum <= 0.0 ? 1.0 : cum;
        for (int i = 0; i < vocab_size; i++) {
            if (probs_scratch[i] < 0.0f)
                probs_scratch[i] = (float)((-probs_scratch[i] - 1e-12f) / kept_sum);
            else
                probs_scratch[i] = 0.0f;
        }
        return multinomial(s, probs_scratch, vocab_size);
    }

    default:
        return -1;
    }
}
