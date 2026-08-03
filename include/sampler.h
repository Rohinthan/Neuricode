#ifndef NEURALC_SAMPLER_H
#define NEURALC_SAMPLER_H

/*
 * sampler.h — sampling strategies over a probability distribution.
 *
 * Deliberately decoupled from Model: sampler_pick() takes a plain
 * float* distribution and knows nothing about tensors, layers, or
 * recurrent state, so it can be reused against any [vocab_size]
 * probability vector, model-produced or not (e.g. unit tests can
 * feed it a hand-built distribution).
 */

#include <stddef.h>

typedef enum {
    SAMPLE_GREEDY      = 0,  /* argmax, temperature/top_k/top_p ignored */
    SAMPLE_TEMPERATURE = 1,  /* rescale in log-space by 1/temperature   */
    SAMPLE_TOP_K       = 2,  /* temperature rescale, then restrict to   *
                              * the top_k highest-probability tokens    */
    SAMPLE_TOP_P       = 3   /* temperature rescale, then restrict to   *
                              * the smallest set whose cumulative prob  *
                              * >= top_p ("nucleus" sampling)           */
} SamplerKind;

typedef struct {
    SamplerKind kind;
    float       temperature; /* used by TEMPERATURE/TOP_K/TOP_P; must   *
                               * be > 0 for those kinds (GREEDY ignores *
                               * it entirely — no epsilon-clamping)     */
    int         top_k;       /* used by TOP_K only; must be >= 1       */
    float       top_p;       /* used by TOP_P only; must be in (0, 1]  */
    float       repetition_penalty; /* penalty for repeated tokens (1.0 = disabled) */
    float       frequency_penalty;  /* penalty for token frequency (0.0 = disabled) */

    /* This sampler's own PRNG stream (rand_r-based), NOT the global
     * rand()/srand(). Initialize via the constructors below, which
     * seed it from a caller-supplied value; sampler_pick() advances it
     * in place on every call. Two Sampler values constructed with the
     * same seed and driven through the same call sequence produce
     * identical output — independent of any other Sampler or thread.
     * Treat this as private state; the field is exposed only because
     * C has no other way to give it storage without a malloc. */
    unsigned    rng_state;
} Sampler;

/* Apply repetition & frequency penalties directly to logits before softmax/sampling */
void sampler_apply_penalties(float *logits, int vocab_size,
                             const int *tokens, int num_tokens,
                             float repetition_penalty, float frequency_penalty);

/* Convenience constructors — fill in sane defaults for the fields the
 * given kind doesn't use, so callers don't have to remember which
 * fields are load-bearing for which kind. */
Sampler sampler_greedy(void);
Sampler sampler_temperature(float temperature, unsigned seed);
Sampler sampler_top_k(int top_k, float temperature, unsigned seed);
Sampler sampler_top_p(float top_p, float temperature, unsigned seed);

/*
 * sampler_pick: selects one token index from `probs` (a probability
 * distribution — NOT logits — of length vocab_size, e.g. straight out
 * of a softmax-activated Dense layer) according to `s->kind`.
 *
 * `logits_scratch` and `probs_scratch` are caller-owned buffers of
 * length vocab_size, reused across calls — no allocation happens
 * inside this function. They may be the same buffers across many
 * calls to the same or different Sampler values.
 *
 * The sampler's own PRNG state (seeded from s->seed on first use per
 * Sampler value — see sampler.c) is used for TEMPERATURE/TOP_K/TOP_P,
 * NOT the global rand()/srand(), so sampling is reproducible and
 * thread-safe across independent Sampler instances.
 *
 * `s` is non-const because TEMPERATURE/TOP_K/TOP_P advance s->rng_state
 * in place (GREEDY never touches it — deterministic by construction).
 *
 * Returns the sampled index in [0, vocab_size), or -1 if `s` is
 * malformed (e.g. top_k <= 0, top_p outside (0,1], temperature <= 0
 * for a kind that requires it) or vocab_size <= 0.
 */
int sampler_pick(Sampler *s, const float *probs, int vocab_size,
                  float *logits_scratch, float *probs_scratch);

#endif /* NEURALC_SAMPLER_H */
