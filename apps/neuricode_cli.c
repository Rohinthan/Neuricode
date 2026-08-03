/*
 * neuricode_cli.c — Antigravity AI Terminal Assistant with 8 Color Themes
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>

#include "tui.h"
#include "pipeline.h"
#include "tokenizer.h"

#define MAX_TOKENS      512
#define MAX_NEW_TOKENS  100
#define INPUT_BYTES     4096
#define MAX_VOCAB       65536

#if defined(NEURALC_HAS_CONFIG) || __has_include("neuralc_config.h")
#include "neuralc_config.h"
#endif

#include "transformer.h"

static TransformerModel *g_trans_model = NULL;

// Hyperparameters
#ifdef NEURALC_TEMPERATURE
static float g_temperature = NEURALC_TEMPERATURE;
#else
static float g_temperature = 0.80f;
#endif
static int   g_top_k       = 20;
static float g_rep_penalty = 1.20f;
static int   g_typing_speed_us = 10000; // 10ms per char

static int file_exists(const char *path) {
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

static void get_exe_directory(char *dir_out, size_t max_sz) {
    dir_out[0] = '\0';
    char path[1024];
    ssize_t count = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (count != -1) {
        path[count] = '\0';
        char *slash = strrchr(path, '/');
        if (slash) {
            *slash = '\0';
            snprintf(dir_out, max_sz, "%s", path);
        }
    }
}

// Auto-discover model and vocab files across standard locations dynamically
static int auto_discover_files(char *model_out, size_t model_sz,
                               char *vocab_out, size_t vocab_sz) {
    const char *env_model = getenv("NEURICODE_MODEL_PATH");
    const char *env_vocab = getenv("NEURICODE_VOCAB_PATH");

    if (env_model && file_exists(env_model)) {
        snprintf(model_out, model_sz, "%s", env_model);
    }
    if (env_vocab && file_exists(env_vocab)) {
        snprintf(vocab_out, vocab_sz, "%s", env_vocab);
    }

    char exe_dir[1024] = "";
    get_exe_directory(exe_dir, sizeof(exe_dir));
    char path_buf[1024];

    if (model_out[0] == '\0') {
        const char *model_candidates[] = {
            "model.bin",
            "assets/model.bin",
            "../model.bin",
            "../assets/model.bin",
            "./assets/model.bin"
        };
        for (size_t i = 0; i < sizeof(model_candidates)/sizeof(model_candidates[0]); i++) {
            if (file_exists(model_candidates[i])) {
                snprintf(model_out, model_sz, "%s", model_candidates[i]);
                break;
            }
            if (exe_dir[0] != '\0') {
                snprintf(path_buf, sizeof(path_buf), "%s/%s", exe_dir, model_candidates[i]);
                if (file_exists(path_buf)) {
                    snprintf(model_out, model_sz, "%s", path_buf);
                    break;
                }
            }
        }
    }

    if (vocab_out[0] == '\0') {
        const char *vocab_candidates[] = {
            "assets/vocab.txt",
            "vocab.txt",
            "../assets/vocab.txt",
            "../vocab.txt",
            "./vocab.txt"
        };
        for (size_t i = 0; i < sizeof(vocab_candidates)/sizeof(vocab_candidates[0]); i++) {
            if (file_exists(vocab_candidates[i])) {
                snprintf(vocab_out, vocab_sz, "%s", vocab_candidates[i]);
                break;
            }
            if (exe_dir[0] != '\0') {
                snprintf(path_buf, sizeof(path_buf), "%s/%s", exe_dir, vocab_candidates[i]);
                if (file_exists(path_buf)) {
                    snprintf(vocab_out, vocab_sz, "%s", path_buf);
                    break;
                }
            }
        }
    }

    return (model_out[0] != '\0' && vocab_out[0] != '\0');
}

static void apply_temperature(float *logits, int vocab, float temp) {
    if (temp < 0.05f) temp = 0.05f;
    for (int i = 0; i < vocab; i++)
        logits[i] /= temp;
}

static void apply_repetition_penalty(float *logits, int vocab, const int *seen, float penalty) {
    for (int i = 0; i < vocab; i++) {
        if (!seen[i]) continue;
        if (logits[i] > 0.0f) logits[i] /= penalty;
        else                  logits[i] *= penalty;
    }
}

static void top_k_filter(float *logits, int vocab, int k) {
    if (k <= 0 || k >= vocab) return;

    static int seen_stamp[MAX_VOCAB];
    static int stamp = 0;
    stamp++;

    for (int pick = 0; pick < k; pick++) {
        float max_val = -1e30f;
        int max_idx = -1;

        for (int i = 0; i < vocab; i++) {
            if (seen_stamp[i] == stamp) continue;
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        if (max_idx >= 0) {
            seen_stamp[max_idx] = stamp;
        }
    }

    for (int i = 0; i < vocab; i++) {
        if (seen_stamp[i] != stamp) {
            logits[i] = -1e30f;
        }
    }
}

static int sample_softmax(const float *logits, int vocab) {
    float max_val = logits[0];
    for (int i = 1; i < vocab; i++)
        if (logits[i] > max_val) max_val = logits[i];

    double sum = 0.0;
    static float probs[MAX_VOCAB];
    for (int i = 0; i < vocab; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    double r = ((double)rand() / (double)RAND_MAX) * sum;
    double acc = 0.0;
    for (int i = 0; i < vocab; i++) {
        acc += probs[i];
        if (acc >= r) return i;
    }
    return vocab - 1;
}

static int generate_next_token(const float *raw_logits, int vocab,
                               const int *seen, float temp, int top_k, float rep_pen) {
    static float work_logits[MAX_VOCAB];
    memcpy(work_logits, raw_logits, (size_t)vocab * sizeof(float));

    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++)
            if (work_logits[i] > work_logits[best]) best = i;
        return best;
    }

    apply_repetition_penalty(work_logits, vocab, seen, rep_pen);
    apply_temperature(work_logits, vocab, temp);
    top_k_filter(work_logits, vocab, top_k);

    return sample_softmax(work_logits, vocab);
}

static void run_generation(Pipeline *pipeline, Tokenizer *tokenizer,
                            int vocab_size, int eos_id,
                            const char *prompt_text,
                            int *tokens, float *logits, int *seen) {
    int num_tokens = tokenizer_encode(tokenizer, prompt_text, tokens, MAX_TOKENS);
    if (num_tokens <= 0) {
        tui_log_warn("Could not parse prompt input into tokens.");
        return;
    }

    tui_show_spinner("Thinking", 250);

    memset(seen, 0, (size_t)vocab_size * sizeof(int));
    for (int i = 0; i < num_tokens; i++)
        seen[tokens[i]] = 1;

#if ACTIVE_MODEL_TYPE == MODEL_TYPE_TRANSFORMER
    if (g_trans_model) {
        for (int i = 0; i < num_tokens; i++) {
            float *res = transformer_forward(g_trans_model, tokens[i], i);
            if (res) memcpy(logits, res, (size_t)vocab_size * sizeof(float));
        }
    } else {
        pipeline_reset(pipeline);
        pipeline_forward(pipeline, tokens, num_tokens, logits);
    }
#else
    pipeline_reset(pipeline);
    pipeline_forward(pipeline, tokens, num_tokens, logits);
#endif
    const float *cur_logits = logits;

    int max_new_tokens = MAX_NEW_TOKENS;
    if (max_new_tokens > MAX_TOKENS - num_tokens)
        max_new_tokens = MAX_TOKENS - num_tokens;

    const Theme *th = tui_get_theme();
    printf("\n%s%sNeuricode >%s ", ANSI_BOLD, th->primary, ANSI_RESET);

    for (int step = 0; step < max_new_tokens; step++) {
        int next_token = generate_next_token(cur_logits, vocab_size, seen,
                                              g_temperature, g_top_k, g_rep_penalty);
        if (next_token < 0 || next_token >= vocab_size) break;

        seen[next_token] = 1;
        tokens[num_tokens++] = next_token;

        char piece[256];
        int written = tokenizer_decode(tokenizer, &next_token, 1, piece, sizeof(piece));
        if (written > 0) {
            char formatted_piece[300];
            if (piece[0] == ' ')
                snprintf(formatted_piece, sizeof(formatted_piece), "%s", piece);
            else
                snprintf(formatted_piece, sizeof(formatted_piece), " %s", piece);

            tui_type_text(formatted_piece, g_typing_speed_us);
        }

        if (next_token == eos_id) break;

#if ACTIVE_MODEL_TYPE == MODEL_TYPE_TRANSFORMER
        if (g_trans_model) {
            float *res = transformer_forward(g_trans_model, next_token, num_tokens - 1);
            if (!res) break;
            cur_logits = res;
        } else {
            cur_logits = pipeline_step(pipeline, next_token);
            if (!cur_logits) break;
        }
#else
        cur_logits = pipeline_step(pipeline, next_token);
        if (!cur_logits) break;
#endif
    }

    printf("\n");
}

static void print_help_menu(void) {
    const Theme *th = tui_get_theme();
    printf("\n%s[NEURICODE COMMANDS]%s\n", th->primary, ANSI_RESET);
    printf("  %s/help%s         - Display help manual\n", th->primary, ANSI_RESET);
    printf("  %s/theme%s        - Open interactive arrow-key theme selector (8 themes)\n", th->primary, ANSI_RESET);
    printf("  %s/status%s       - Display live system status & parameters\n", th->primary, ANSI_RESET);
    printf("  %s/temp <val>%s   - Set sampling temperature (current: %.2f)\n", th->primary, ANSI_RESET, g_temperature);
    printf("  %s/topk <val>%s   - Set Top-K sampling cap (current: %d)\n", th->primary, ANSI_RESET, g_top_k);
    printf("  %s/log%s          - Test colored status logs\n", th->primary, ANSI_RESET);
    printf("  %s/reset%s        - Reset hidden states & KV cache\n", th->primary, ANSI_RESET);
    printf("  %s/clear%s        - Clear terminal screen\n", th->primary, ANSI_RESET);
    printf("  %s/exit%s         - Quit Neuricode CLI\n", th->primary, ANSI_RESET);
}

static void print_status_dashboard(const char *model_path, const char *vocab_path) {
    const Theme *th = tui_get_theme();
    printf("\n%s[SYSTEM DASHBOARD]%s\n", th->primary, ANSI_RESET);
#if ACTIVE_MODEL_TYPE == MODEL_TYPE_TRANSFORMER
    printf("  Engine      : \033[36mTransformer Engine (Self-Attention + KV-Cache)\033[0m\n");
#else
    printf("  Engine      : \033[36mRecurrent Neural Network (RNN)\033[0m\n");
#endif
    printf("  Model File  : %s\n", model_path);
    printf("  Vocab File  : %s\n", vocab_path);
    printf("  Theme       : %s%s%s\n", th->primary, th->name, ANSI_RESET);
    printf("  Temperature : %.2f\n", g_temperature);
    printf("  Top-K       : %d\n", g_top_k);
    printf("  Rep Penalty : %.2f\n", g_rep_penalty);
    printf("  Status      : \033[32mACTIVE & OPERATIONAL\033[0m\n");
}

static void handle_theme_selection(void) {
    const char *options[] = {
        "Classic Purple   (Default)",
        "Cyberpunk Cyan   (Cyan Accent)",
        "Matrix Green     (Electric Green)",
        "Sunset Orange    (Neon Orange)",
        "Electric Blue    (Cobalt Blue)",
        "Crimson Red      (Neon Crimson)",
        "Emerald Mint     (Teal / Mint)",
        "Monochrome Dark  (Slate Gray / Silver)"
    };
    int selected = tui_interactive_menu("Select Palette", options, 8);
    if (selected >= 0 && selected < 8) {
        tui_set_theme((ThemeID)selected);
        tui_log_success("Theme updated to: %s", tui_get_theme_name((ThemeID)selected));
    }
}

static void demo_colored_logs(void) {
    printf("\n");
    tui_log_info("Initializing Neural Network pipeline engine...");
    tui_log_warn("High memory allocation detected in KV cache.");
    tui_log_error("Sample error message: Model parameter out of bounds.");
    tui_log_success("All system components validated cleanly!");
}

int main(int argc, char **argv) {
    tui_init();
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)&argc);

    char model_path[1024] = "";
    char vocab_path[1024] = "";

    if (argc >= 3) {
        snprintf(model_path, sizeof(model_path), "%s", argv[1]);
        snprintf(vocab_path, sizeof(vocab_path), "%s", argv[2]);
    } else {
        if (!auto_discover_files(model_path, sizeof(model_path), vocab_path, sizeof(vocab_path))) {
            tui_log_error("Model file (model.bin) or Vocab file (assets/vocab.txt) not found.");
            fprintf(stderr, "Usage: neuricode [model.bin] [assets/vocab.txt]\n");
            return 1;
        }
    }

    tui_clear_screen();
    tui_print_header_banner("v1.0", model_path);

#if ACTIVE_MODEL_TYPE == MODEL_TYPE_TRANSFORMER
    g_trans_model = transformer_init_from_config();
    if (g_trans_model) {
        tui_log_success("Loaded Native Transformer Engine (vocab=%d, dim=%d, layers=%d, heads=%d)",
                        g_trans_model->config.vocab_size, g_trans_model->config.dim,
                        g_trans_model->config.n_layers, g_trans_model->config.n_heads);
    }
#endif

    Pipeline *pipeline = pipeline_load(model_path);
    if (!pipeline) {
        tui_log_error("Could not load model from '%s'", model_path);
        if (g_trans_model) transformer_free(g_trans_model);
        return 1;
    }

    Tokenizer *tokenizer = tokenizer_load(vocab_path);
    if (!tokenizer) {
        tui_log_error("Could not load tokenizer from '%s'", vocab_path);
        if (g_trans_model) transformer_free(g_trans_model);
        return 1;
    }

    int vocab_size = tokenizer_vocab_size(tokenizer);
    int eos_id = tokenizer_eos_id(tokenizer);

    int   *tokens = calloc(MAX_TOKENS, sizeof(*tokens));
    float *logits = calloc((size_t)vocab_size, sizeof(*logits));
    int   *seen   = calloc((size_t)vocab_size, sizeof(*seen));

    if (!tokens || !logits || !seen) {
        tui_log_error("Out of memory allocation failure.");
        if (g_trans_model) transformer_free(g_trans_model);
        return 1;
    }

    char input[INPUT_BYTES];
    while (1) {
        tui_print_antigravity_input_frame_start();

        if (!fgets(input, sizeof(input), stdin)) break;

        tui_print_antigravity_input_frame_end();

        input[strcspn(input, "\r\n")] = '\0';
        if (input[0] == '\0') continue;

        // Handle slash commands
        if (strcmp(input, "/exit") == 0 || strcmp(input, "/quit") == 0) {
            tui_log_info("Goodbye! Session closed.");
            break;
        } else if (strcmp(input, "/help") == 0) {
            print_help_menu();
            continue;
        } else if (strcmp(input, "/theme") == 0) {
            handle_theme_selection();
            continue;
        } else if (strcmp(input, "/status") == 0) {
            print_status_dashboard(model_path, vocab_path);
            continue;
        } else if (strcmp(input, "/log") == 0) {
            demo_colored_logs();
            continue;
        } else if (strcmp(input, "/clear") == 0) {
            tui_clear_screen();
            tui_print_header_banner("v1.0", model_path);
            continue;
        } else if (strncmp(input, "/temp ", 6) == 0) {
            float t = strtof(input + 6, NULL);
            if (t > 0.0f) {
                g_temperature = t;
                tui_log_success("Temperature updated to %.2f", g_temperature);
            } else {
                tui_log_warn("Invalid temperature value.");
            }
            continue;
        } else if (strncmp(input, "/topk ", 6) == 0) {
            int k = atoi(input + 6);
            if (k > 0) {
                g_top_k = k;
                tui_log_success("Top-K updated to %d", g_top_k);
            } else {
                tui_log_warn("Invalid Top-K value.");
            }
            continue;
        } else if (strcmp(input, "/reset") == 0) {
            pipeline_reset(pipeline);
            tui_log_success("Model hidden states and memory reset.");
            continue;
        }

        // Code highlight fallback
        if (strncmp(input, "code", 4) == 0 || strncmp(input, "def ", 4) == 0 || strncmp(input, "int ", 4) == 0) {
            const char *sample_code =
                "int main(void) {\n"
                "    printf(\"Hello, Neuricode Antigravity CLI!\\n\");\n"
                "    return 0;\n"
                "}";
            tui_print_highlighted_code(sample_code);
            continue;
        }

        run_generation(pipeline, tokenizer, vocab_size, eos_id, input, tokens, logits, seen);
    }

    free(tokens);
    free(logits);
    free(seen);
    tokenizer_free(tokenizer);
    if (g_trans_model) transformer_free(g_trans_model);
    return 0;
}
