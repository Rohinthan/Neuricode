#ifndef PIPELINE_H
#define PIPELINE_H

typedef struct Pipeline Pipeline;

Pipeline *pipeline_load(const char *model_path);

/*
 * pipeline_forward: runs the full sequence through the model and
 * copies the LAST timestep's logits into `output` (caller-allocated,
 * size >= vocab_size). Unchanged — still the entry point for feeding
 * a multi-token prompt.
 */
void pipeline_forward(Pipeline *p, const int *input, int seq_len, float *output);

/*
 * pipeline_step: single-token inference. Advances the model's hidden
 * state by exactly one token and returns a pointer to that step's
 * logits.
 *
 * The returned pointer is owned by `p` (points into an internal,
 * already-allocated buffer reused on every call) — do NOT free it,
 * and treat it as valid only until the next pipeline_step()/
 * pipeline_forward()/pipeline_reset() call on the same Pipeline.
 *
 * Returns NULL if `p` is NULL or token_id is outside [0, vocab_size).
 */
const float *pipeline_step(Pipeline *p, int token_id);

void pipeline_reset(Pipeline *p);
void pipeline_free(Pipeline *p);

#endif /* PIPELINE_H */
