#ifndef TOKENIZER_H
#define TOKENIZER_H

/*
 * tokenizer.h — minimal SentencePiece-style tokenizer for a
 * llama.cpp-style inference pipeline.
 *
 * ── Vocabulary format ───────────────────────────────────────────────
 * A plain text file, one token per line. Token ID == line index
 * (0-based). Lines are taken verbatim (minus the trailing newline) —
 * if your dump already represents spaces-within-tokens as the
 * SentencePiece marker "▁" (U+2581, UTF-8 bytes E2 96 81), that's
 * exactly what this loader expects; it does no transformation of
 * vocab entries themselves.
 *
 * ── Algorithm ───────────────────────────────────────────────────────
 * Encoding is greedy longest-match, left to right: at each position,
 * try the longest substring (bounded by the longest token in the
 * vocab) that exists in the vocab, emit its ID, advance past it, and
 * repeat. No BPE merge ranks, no unicode normalization — exactly the
 * "keep it simple" scope this is meant to cover.
 *
 * Before matching, literal ASCII spaces in the input are rewritten to
 * the SentencePiece marker "▁" (this is what lets a vocab entry like
 * "▁hello" match "the word after a space"), and — matching
 * SentencePiece's default add_dummy_prefix behavior — a marker is
 * also prepended if the text doesn't already start with a space, so
 * the first word tokenizes the same way as an interior word.
 *
 * If no vocab entry matches even a single byte at some position,
 * encoding falls back, in order:
 *   1. a byte-fallback token formatted "<0x%02X>" (the convention
 *      real SentencePiece/llama.cpp vocabularies use for this), if
 *      present in the vocab;
 *   2. an "<unk>" token, if present in the vocab;
 *   3. otherwise the byte is dropped (a warning is printed once per
 *      load if this path is ever hit) — the byte is still consumed,
 *      so encoding always makes forward progress and terminates.
 *
 * ── Special tokens ───────────────────────────────────────────────────
 * "<unk>" is recognized by exact string match if present anywhere in
 * the vocab. EOS is recognized the same way, accepting either "</s>"
 * (SentencePiece's convention) or "<eos>" — whichever is found first
 * by line order — since this project doesn't mandate one spelling. A
 * vocab file with neither is fully supported: tokenizer_eos_id()
 * returns TOKENIZER_INVALID_ID rather than guessing a fallback id, so
 * "no EOS declared" can never be mistaken for a real token.
 *
 * ── Ownership / thread-safety ───────────────────────────────────────
 * A Tokenizer is read-only after tokenizer_load() returns, so a
 * single instance may be safely shared and used concurrently by
 * multiple threads calling tokenizer_encode()/tokenizer_decode() (both
 * take a `const Tokenizer *`  and touch no mutable state).
 */

#include <stddef.h>

#define TOKENIZER_INVALID_ID (-1)

typedef struct Tokenizer Tokenizer;

/*
 * tokenizer_load: reads and validates the vocab file at `vocab_path`
 * (one token per line, ID = line index), building an in-memory
 * id->string table plus a hash index for string->id lookups.
 *
 * Returns a fully populated Tokenizer* on success. Returns NULL on
 * any failure (file not found, empty vocab, a line exceeding the
 * internal per-line sanity limit, or allocation failure) — everything
 * allocated so far is freed before returning, so there is nothing for
 * the caller to clean up on failure.
 */
Tokenizer *tokenizer_load(const char *vocab_path);

/* Frees every buffer owned by `t`. Safe to call with t == NULL. */
void tokenizer_free(Tokenizer *t);

/* Number of entries in the loaded vocabulary (== number of lines in
 * the vocab file). */
int tokenizer_vocab_size(const Tokenizer *t);

/*
 * tokenizer_eos_id: id of the EOS token ("</s>" or "<eos>"), found by
 * exact string match during tokenizer_load() — same mechanism as the
 * existing "<unk>" detection. Returns TOKENIZER_INVALID_ID (-1) if
 * the loaded vocab declares neither spelling, or if t is NULL. -1 is
 * never a valid token id, so callers (e.g. cli.c's
 * `next_token == eos_id` stop check) can compare against it
 * unconditionally without a separate "does this tokenizer have an
 * EOS token" branch.
 */
int tokenizer_eos_id(const Tokenizer *t);

/*
 * tokenizer_encode: greedy longest-match tokenization of `text` (a
 * NUL-terminated string) into `tokens` (caller-allocated, capacity
 * `max_tokens`).
 *
 * If `text` needs more than `max_tokens` tokens, encoding stops early
 * — `tokens` holds a valid prefix of the full tokenization, never a
 * partial/garbage token, and the returned count reflects exactly what
 * was written.
 *
 * Returns the number of tokens written (0..max_tokens), or -1 if
 * `t`/`text`/`tokens` is NULL or max_tokens <= 0.
 */
int tokenizer_encode(const Tokenizer *t, const char *text,
                     int *tokens, int max_tokens);

/*
 * tokenizer_decode: renders `num_tokens` token IDs back to text,
 * reversing the space<->marker substitution from tokenizer_encode(),
 * writing at most `output_size - 1` bytes into `output` plus a NUL
 * terminator (always NUL-terminated if output_size > 0).
 *
 * Returns the number of bytes written (excluding the NUL terminator,
 * possibly less than the full untruncated decode if output_size was
 * too small), or -1 if `t`/`tokens`/`output` is NULL, output_size == 0,
 * or any id in `tokens` is out of range for this vocabulary.
 */
int tokenizer_decode(const Tokenizer *t, const int *tokens, int num_tokens,
                     char *output, size_t output_size);

#endif /* TOKENIZER_H */
