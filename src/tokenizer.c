/*
 * tokenizer.c — see tokenizer.h for the format/algorithm contract.
 *
 * No external libraries: only the C standard library. The only
 * "cleverness" here is a small open-addressing hash table for
 * string->id lookups, so greedy longest-match encoding doesn't
 * degrade to an O(vocab_size) linear scan at every position.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "tokenizer.h"

/* SentencePiece's word-boundary marker, U+2581 "▁", UTF-8 encoded. */
static const char SPACE_MARKER[3]  = { (char)0xE2, (char)0x96, (char)0x81 };
#define SPACE_MARKER_LEN 3

/* Hard ceiling on a single vocab line's length — guards against a
 * corrupt/absurd vocab file the same way the model loader guards
 * against corrupt dimensions: reject cleanly instead of silently
 * truncating a token's text or looping past a too-small buffer. */
#define MAX_LINE_LEN 4096

/* ═══════════════════════════════════════════════════════════════════
 *  Hash table — open addressing, linear probing, FNV-1a over the
 *  exact byte range being looked up (no temporary allocation needed
 *  to hash or compare a substring).
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *token;   /* NULL = empty slot; else points into
                           * Tokenizer.vocab[...] (not separately owned) */
    int         id;
} HashSlot;

struct Tokenizer {
    char   **vocab;         /* vocab[id] = token string, owned, NUL-terminated */
    int      vocab_size;
    int      max_token_len; /* longest single vocab entry, in bytes */

    HashSlot *table;
    size_t    table_cap;    /* always a power of two */

    int      unk_id;        /* id of "<unk>" if present, else TOKENIZER_INVALID_ID */
    int      eos_id;        /* id of "</s>" or "<eos>" (first found, in that
                              * scan order) if present, else TOKENIZER_INVALID_ID */
};

static uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* Insert (token,id) into the table. `token` must outlive the table
 * (it points into Tokenizer.vocab). Table must have free slots. */
static void hash_insert(HashSlot *table, size_t cap, const char *token, int id) {
    size_t len = strlen(token);
    uint32_t h = fnv1a(token, len);
    size_t idx = h & (cap - 1);
    while (table[idx].token != NULL) {
        idx = (idx + 1) & (cap - 1);
    }
    table[idx].token = token;
    table[idx].id    = id;
}

/* Looks up the exact byte range s[0..len) as a whole token. Returns
 * its id, or TOKENIZER_INVALID_ID if no vocab entry matches. */
static int hash_lookup(const Tokenizer *t, const char *s, size_t len) {
    uint32_t h = fnv1a(s, len);
    size_t idx = h & (t->table_cap - 1);
    size_t probes = 0;
    while (t->table[idx].token != NULL && probes < t->table_cap) {
        const char *cand = t->table[idx].token;
        if (strlen(cand) == len && memcmp(cand, s, len) == 0)
            return t->table[idx].id;
        idx = (idx + 1) & (t->table_cap - 1);
        probes++;
    }
    return TOKENIZER_INVALID_ID;
}

/* ═══════════════════════════════════════════════════════════════════
 *  tokenizer_load / tokenizer_free
 * ═══════════════════════════════════════════════════════════════════ */

static char *dup_line(const char *s, size_t len) {
    char *d = (char *)malloc(len + 1);
    if (!d) return NULL;
    memcpy(d, s, len);
    d[len] = '\0';
    return d;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++)
            free(t->vocab[i]);
        free(t->vocab);
    }
    free(t->table);
    free(t);
}

Tokenizer *tokenizer_load(const char *vocab_path) {
    if (!vocab_path) {
        fprintf(stderr, "tokenizer_load: null vocab_path\n");
        return NULL;
    }

    FILE *f = fopen(vocab_path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer_load: cannot open '%s'\n", vocab_path);
        return NULL;
    }

    Tokenizer *t = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    if (!t) {
        fprintf(stderr, "tokenizer_load: out of memory\n");
        fclose(f);
        return NULL;
    }
    t->unk_id = TOKENIZER_INVALID_ID;
    t->eos_id = TOKENIZER_INVALID_ID;

    size_t cap = 4096;
    t->vocab = (char **)malloc(cap * sizeof(char *));
    if (!t->vocab) {
        fprintf(stderr, "tokenizer_load: out of memory\n");
        goto fail;
    }

    char line[MAX_LINE_LEN];
    int line_no = 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);

        /* A full line ends with '\n' (or is the last line in a file
         * with no trailing newline, which fgets also returns intact
         * as long as it fit). If the buffer filled completely without
         * seeing '\n' and we're not at EOF, the line was longer than
         * MAX_LINE_LEN — reject the file rather than silently
         * truncating a token's text. */
        if (len == sizeof(line) - 1 && line[len - 1] != '\n' && !feof(f)) {
            fprintf(stderr,
                "tokenizer_load: line %d exceeds max line length (%d bytes)\n",
                line_no + 1, MAX_LINE_LEN);
            goto fail;
        }

        /* strip trailing \n and \r (in case of CRLF vocab files) */
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            len--;

        if ((size_t)t->vocab_size >= cap) {
            size_t new_cap = cap * 2;
            char **grown = (char **)realloc(t->vocab, new_cap * sizeof(char *));
            if (!grown) {
                fprintf(stderr, "tokenizer_load: out of memory\n");
                goto fail;
            }
            t->vocab = grown;
            cap = new_cap;
        }

        char *tok = dup_line(line, len);
        if (!tok) {
            fprintf(stderr, "tokenizer_load: out of memory\n");
            goto fail;
        }
        t->vocab[t->vocab_size] = tok;

        if ((int)len > t->max_token_len)
            t->max_token_len = (int)len;
        if (t->unk_id == TOKENIZER_INVALID_ID && strcmp(tok, "<unk>") == 0)
            t->unk_id = t->vocab_size;
        if (t->eos_id == TOKENIZER_INVALID_ID &&
            (strcmp(tok, "</s>") == 0 || strcmp(tok, "<eos>") == 0))
            t->eos_id = t->vocab_size;

        t->vocab_size++;
        line_no++;
    }
    fclose(f);
    f = NULL;

    if (t->vocab_size == 0) {
        fprintf(stderr, "tokenizer_load: vocab file '%s' is empty\n", vocab_path);
        goto fail;
    }

    /* Build the hash index — load factor <= 0.5 for short probe chains. */
    t->table_cap = next_pow2((size_t)t->vocab_size * 2);
    t->table = (HashSlot *)calloc(t->table_cap, sizeof(HashSlot));
    if (!t->table) {
        fprintf(stderr, "tokenizer_load: out of memory\n");
        goto fail;
    }
    for (int i = 0; i < t->vocab_size; i++)
        hash_insert(t->table, t->table_cap, t->vocab[i], i);

    return t;

fail:
    if (f) fclose(f);
    tokenizer_free(t);
    return NULL;
}

int tokenizer_vocab_size(const Tokenizer *t) {
    return t ? t->vocab_size : 0;
}

/* id of the EOS token ("</s>" or "<eos>", whichever appears in the
 * vocab — see the struct comment), or TOKENIZER_INVALID_ID if neither
 * is present. Deliberately does NOT fall back to "last vocab line" —
 * an undeclared EOS means "no stop token", not "guess one"; a wrong
 * guess would silently truncate generation on an ordinary content
 * token. cli.c's `next_token == eos_id` check is already safe against
 * TOKENIZER_INVALID_ID (-1), since no real token id is ever negative. */
int tokenizer_eos_id(const Tokenizer *t) {
    return t ? t->eos_id : TOKENIZER_INVALID_ID;
}

/* ═══════════════════════════════════════════════════════════════════
 *  tokenizer_encode
 * ═══════════════════════════════════════════════════════════════════ */

/* Emits a single-byte fallback token for buf[pos], per the priority
 * order documented in tokenizer.h. Always returns a valid id to
 * emit, OR TOKENIZER_INVALID_ID if the byte must simply be dropped
 * (still consumes exactly 1 byte from the caller's perspective either
 * way — the caller advances pos regardless of this return value). */
static int fallback_byte(const Tokenizer *t, unsigned char byte) {
    char hex[8];
    snprintf(hex, sizeof(hex), "<0x%02X>", byte);
    int id = hash_lookup(t, hex, strlen(hex));
    if (id != TOKENIZER_INVALID_ID) return id;

    if (t->unk_id != TOKENIZER_INVALID_ID) return t->unk_id;

    return TOKENIZER_INVALID_ID;
}

int tokenizer_encode(const Tokenizer *t, const char *text,
                     int *tokens, int max_tokens) {
    if (!t || !text || !tokens || max_tokens <= 0) return -1;

    size_t textlen = strlen(text);

    /* Worst case: every input byte becomes a 3-byte marker, plus one
     * possible leading marker, plus the NUL terminator. */
    size_t buf_cap = textlen * SPACE_MARKER_LEN + SPACE_MARKER_LEN + 1;
    char *buf = (char *)malloc(buf_cap);
    if (!buf) return -1;

    size_t bp = 0;
    /* add_dummy_prefix: treat start-of-text as a word boundary too,
     * same as SentencePiece's default, so the first word tokenizes
     * identically to an interior word. */
    if (textlen > 0 && text[0] != ' ') {
        memcpy(buf + bp, SPACE_MARKER, SPACE_MARKER_LEN);
        bp += SPACE_MARKER_LEN;
    }
    for (size_t i = 0; i < textlen; i++) {
        if (text[i] == ' ') {
            memcpy(buf + bp, SPACE_MARKER, SPACE_MARKER_LEN);
            bp += SPACE_MARKER_LEN;
        } else {
            buf[bp++] = text[i];
        }
    }
    size_t buflen = bp;

    int out_count = 0;
    size_t pos = 0;
    while (pos < buflen && out_count < max_tokens) {
        size_t max_len = buflen - pos;
        if ((size_t)t->max_token_len < max_len) max_len = (size_t)t->max_token_len;

        int matched_id  = TOKENIZER_INVALID_ID;
        size_t matched_len = 0;

        for (size_t L = max_len; L > 0; L--) {
            int id = hash_lookup(t, buf + pos, L);
            if (id != TOKENIZER_INVALID_ID) {
                matched_id  = id;
                matched_len = L;
                break;
            }
        }

        if (matched_id != TOKENIZER_INVALID_ID) {
            tokens[out_count++] = matched_id;
            pos += matched_len;
        } else {
            /* No vocab entry matches even a single byte here — fall
             * back per tokenizer.h's documented priority order. Always
             * advance by exactly 1 byte so this loop is guaranteed to
             * terminate regardless of vocab contents. */
            int fb = fallback_byte(t, (unsigned char)buf[pos]);
            if (fb != TOKENIZER_INVALID_ID)
                tokens[out_count++] = fb;
            pos += 1;
        }
    }

    free(buf);
    return out_count;
}

/* ═══════════════════════════════════════════════════════════════════
 *  tokenizer_decode
 * ═══════════════════════════════════════════════════════════════════ */

int tokenizer_decode(const Tokenizer *t, const int *tokens, int num_tokens,
                     char *output, size_t output_size) {
    if (!t || !tokens || !output || output_size == 0) return -1;
    if (num_tokens < 0) return -1;

    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= t->vocab_size) {
            output[0] = '\0';
            return -1;
        }
    }

    size_t written = 0;
    for (int i = 0; i < num_tokens && written + 1 < output_size; i++) {
        const char *s = t->vocab[tokens[i]];
        size_t slen = strlen(s);
        size_t p = 0;
        while (p < slen && written + 1 < output_size) {
            if (p + SPACE_MARKER_LEN <= slen &&
                memcmp(s + p, SPACE_MARKER, SPACE_MARKER_LEN) == 0) {
                output[written++] = ' ';
                p += SPACE_MARKER_LEN;
            } else {
                output[written++] = s[p];
                p += 1;
            }
        }
    }

    output[written] = '\0';
    return (int)written;
}
