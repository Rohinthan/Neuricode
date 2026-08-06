/*
 * tui.c — Expanded Color Themes Engine for Neuricode CLI
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <termios.h>
#include <time.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <ctype.h>
#include "tui.h"

// ═══════════════════════════════════════════════════════════════════
//  8 COLOR THEME PALETTES DEFINITIONS
// ═══════════════════════════════════════════════════════════════════

static Theme g_themes[THEME_COUNT] = {
    [THEME_PURPLE_CLASSIC] = {
        .name      = "Classic Purple (Default)",
        .primary   = NEON_PURPLE,
        .prompt    = NEON_PURPLE,
        .text      = "\033[37m",
        .muted     = NEON_PURPLE
    },
    [THEME_CYBERPUNK] = {
        .name      = "Cyberpunk Cyan",
        .primary   = NEON_CYAN,
        .prompt    = NEON_CYAN,
        .text      = "\033[37m",
        .muted     = NEON_CYAN
    },
    [THEME_MATRIX] = {
        .name      = "Matrix Green",
        .primary   = NEON_GREEN,
        .prompt    = NEON_GREEN,
        .text      = "\033[37m",
        .muted     = NEON_GREEN
    },
    [THEME_SUNSET_ORANGE] = {
        .name      = "Sunset Orange",
        .primary   = NEON_ORANGE,
        .prompt    = NEON_ORANGE,
        .text      = "\033[37m",
        .muted     = NEON_ORANGE
    },
    [THEME_ELECTRIC_BLUE] = {
        .name      = "Electric Blue",
        .primary   = NEON_BLUE,
        .prompt    = NEON_BLUE,
        .text      = "\033[37m",
        .muted     = NEON_BLUE
    },
    [THEME_CRIMSON_RED] = {
        .name      = "Crimson Red",
        .primary   = NEON_RED,
        .prompt    = NEON_RED,
        .text      = "\033[37m",
        .muted     = NEON_RED
    },
    [THEME_EMERALD_MINT] = {
        .name      = "Emerald Mint",
        .primary   = NEON_MINT,
        .prompt    = NEON_MINT,
        .text      = "\033[37m",
        .muted     = NEON_MINT
    },
    [THEME_MONOCHROME_DARK] = {
        .name      = "Monochrome Dark",
        .primary   = COLOR_SILVER,
        .prompt    = COLOR_SILVER,
        .text      = "\033[37m",
        .muted     = COLOR_DARK_GRAY
    }
};

static ThemeID g_current_theme_id = THEME_PURPLE_CLASSIC;
static struct termios g_orig_termios;
static bool g_raw_mode_active = false;
static int g_cached_width = 0;

static const char *get_theme_config_path(char *buf, size_t sz) {
    const char *home = getenv("HOME");
    if (!home) home = ".";
    snprintf(buf, sz, "%s/.neuricode_theme", home);
    return buf;
}

void tui_load_theme_config(void) {
    char path[1024];
    get_theme_config_path(path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (f) {
        int id = 0;
        if (fscanf(f, "%d", &id) == 1) {
            if (id >= 0 && id < THEME_COUNT) {
                g_current_theme_id = (ThemeID)id;
            }
        }
        fclose(f);
    }
}

void tui_save_theme_config(ThemeID id) {
    if (id >= THEME_COUNT) return;
    char path[1024];
    get_theme_config_path(path, sizeof(path));
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "%d\n", (int)id);
        fclose(f);
    }
}

const Theme *tui_get_theme(void) {
    return &g_themes[g_current_theme_id];
}

void tui_set_theme(ThemeID id) {
    if (id < THEME_COUNT) {
        g_current_theme_id = id;
        tui_save_theme_config(id);
    }
}

const char *tui_get_theme_name(ThemeID id) {
    if (id < THEME_COUNT) return g_themes[id].name;
    return "Unknown";
}

// ═══════════════════════════════════════════════════════════════════
//  DYNAMIC TERMINAL WINDOW RESIZING (SIGWINCH)
// ═══════════════════════════════════════════════════════════════════

void tui_update_terminal_size(void) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 15) {
        g_cached_width = w.ws_col;
        return;
    }
    const char *cols_env = getenv("COLUMNS");
    if (cols_env) {
        int cols = atoi(cols_env);
        if (cols > 15) {
            g_cached_width = cols;
            return;
        }
    }
    g_cached_width = 80;
}

static void handle_sigwinch(int sig) {
    (void)sig;
    tui_update_terminal_size();
}

void tui_init(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    tui_load_theme_config();
    tui_update_terminal_size();
    signal(SIGWINCH, handle_sigwinch);
}

void tui_clear_screen(void) {
    printf("\033[2J\033[H");
    fflush(stdout);
}

int tui_get_terminal_width(void) {
    if (g_cached_width <= 15) {
        tui_update_terminal_size();
    }
    return g_cached_width;
}

void tui_enable_raw_mode(void) {
    if (g_raw_mode_active) return;
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    g_raw_mode_active = true;
}

void tui_disable_raw_mode(void) {
    if (!g_raw_mode_active) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    g_raw_mode_active = false;
}

static void sleep_us(long us) {
    struct timespec req;
    req.tv_sec = us / 1000000L;
    req.tv_nsec = (us % 1000000L) * 1000L;
    nanosleep(&req, NULL);
}

// ═══════════════════════════════════════════════════════════════════
//  CLASSIC ANTIGRAVITY HEADER & STRAIGHT INPUT LINES
// ═══════════════════════════════════════════════════════════════════

void tui_print_header_banner(const char *version, const char *model_name) {
    const Theme *th = tui_get_theme();
    int width = tui_get_terminal_width();

    printf("\n");
    printf("%s%s         /\\_/\\         %s\n", ANSI_BOLD, NEON_CYAN, ANSI_RESET);
    printf("%s%s        ( o.o )        %s\n", ANSI_BOLD, NEON_CYAN, ANSI_RESET);
    printf("%s%s         > ^ <         %s\n", ANSI_BOLD, NEON_CYAN, ANSI_RESET);
    printf("\n");
    printf("     %s%sNeuricode CLI 1.1.0%s\n", ANSI_BOLD, th->primary, ANSI_RESET);
    printf("\n");

    printf("%s", th->muted);
    for (int i = 0; i < width; i++) printf("─");
    printf("%s\n", ANSI_RESET);

    printf(" %s%sNEURICODE CLI%s %s[%s]%s %s• Model: %s • Type /help for commands%s\n",
           th->primary, ANSI_BOLD, ANSI_RESET,
           th->primary, version ? version : "1.1.0", ANSI_RESET,
           COLOR_DARK_GRAY, model_name ? model_name : "model.bin", ANSI_RESET);

    printf("%s", th->muted);
    for (int i = 0; i < width; i++) printf("─");
    printf("%s\n\n", ANSI_RESET);
}

#define MAX_HIST_ENTRIES 200
static char g_cmd_history[MAX_HIST_ENTRIES][1024];
static int g_cmd_history_count = 0;

bool tui_readline(const char *prompt_prefix, char *out_buf, size_t max_len) {
    if (!out_buf || max_len == 0) return false;
    out_buf[0] = '\0';

    if (!isatty(STDIN_FILENO)) {
        if (!fgets(out_buf, (int)max_len, stdin)) return false;
        out_buf[strcspn(out_buf, "\r\n")] = '\0';
        return true;
    }

    tui_enable_raw_mode();

    char line[1024] = "";
    int len = 0;
    int pos = 0;
    int hist_idx = g_cmd_history_count;

    if (prompt_prefix && prompt_prefix[0] != '\0') {
        printf("%s", prompt_prefix);
        fflush(stdout);
    }

    while (1) {
        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) {
            tui_disable_raw_mode();
            return false;
        }

        if (c == '\r' || c == '\n') {
            printf("\n");
            fflush(stdout);
            break;
        } else if (c == 4) { // Ctrl+D
            if (len == 0) {
                tui_disable_raw_mode();
                return false;
            }
        } else if (c == 127 || c == '\b') { // Backspace
            if (pos > 0) {
                memmove(&line[pos - 1], &line[pos], len - pos + 1);
                pos--;
                len--;
                printf("\r%s%s\033[K", prompt_prefix ? prompt_prefix : "", line);
                if (len - pos > 0) {
                    printf("\033[%dD", len - pos);
                }
                fflush(stdout);
            }
        } else if (c == '\033') { // Escape sequences (Arrow keys)
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[0] == '[') {
                    if (seq[1] == 'A') { // UP Arrow
                        if (g_cmd_history_count > 0 && hist_idx > 0) {
                            hist_idx--;
                            strncpy(line, g_cmd_history[hist_idx], sizeof(line) - 1);
                            line[sizeof(line) - 1] = '\0';
                            len = (int)strlen(line);
                            pos = len;
                            printf("\r%s%s\033[K", prompt_prefix ? prompt_prefix : "", line);
                            fflush(stdout);
                        }
                    } else if (seq[1] == 'B') { // DOWN Arrow
                        if (hist_idx < g_cmd_history_count) {
                            hist_idx++;
                            if (hist_idx == g_cmd_history_count) {
                                line[0] = '\0';
                            } else {
                                strncpy(line, g_cmd_history[hist_idx], sizeof(line) - 1);
                                line[sizeof(line) - 1] = '\0';
                            }
                            len = (int)strlen(line);
                            pos = len;
                            printf("\r%s%s\033[K", prompt_prefix ? prompt_prefix : "", line);
                            fflush(stdout);
                        }
                    } else if (seq[1] == 'D') { // LEFT Arrow
                        if (pos > 0) {
                            pos--;
                            printf("\033[D");
                            fflush(stdout);
                        }
                    } else if (seq[1] == 'C') { // RIGHT Arrow
                        if (pos < len) {
                            pos++;
                            printf("\033[C");
                            fflush(stdout);
                        }
                    }
                }
            }
        } else if (isprint((unsigned char)c)) {
            if (len < (int)sizeof(line) - 1 && len < (int)max_len - 1) {
                memmove(&line[pos + 1], &line[pos], len - pos + 1);
                line[pos] = c;
                pos++;
                len++;
                printf("\r%s%s", prompt_prefix ? prompt_prefix : "", line);
                if (len - pos > 0) {
                    printf("\033[%dD", len - pos);
                }
                fflush(stdout);
            }
        }
    }

    tui_disable_raw_mode();
    strncpy(out_buf, line, max_len - 1);
    out_buf[max_len - 1] = '\0';

    if (len > 0) {
        if (g_cmd_history_count == 0 || strcmp(g_cmd_history[g_cmd_history_count - 1], line) != 0) {
            if (g_cmd_history_count < MAX_HIST_ENTRIES) {
                snprintf(g_cmd_history[g_cmd_history_count], 1024, "%s", line);
                g_cmd_history_count++;
            } else {
                for (int i = 0; i < MAX_HIST_ENTRIES - 1; i++) {
                    strcpy(g_cmd_history[i], g_cmd_history[i + 1]);
                }
                snprintf(g_cmd_history[MAX_HIST_ENTRIES - 1], 1024, "%s", line);
            }
        }
    }

    return true;
}

void tui_print_antigravity_input_frame_start(void) {
    const Theme *th = tui_get_theme();
    int width = tui_get_terminal_width();

    printf("\n%s", th->muted);
    for (int i = 0; i < width; i++) printf("─");
    printf("%s\n", ANSI_RESET);

    printf("%s%s> %s", ANSI_BOLD, th->prompt, ANSI_RESET);
    fflush(stdout);
}

void tui_print_antigravity_input_frame_end(void) {
    const Theme *th = tui_get_theme();
    int width = tui_get_terminal_width();

    printf("%s", th->muted);
    for (int i = 0; i < width; i++) printf("─");
    printf("%s\n", ANSI_RESET);
    fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════
//  LOGGING & STATUS BADGES
// ═══════════════════════════════════════════════════════════════════

void tui_log_info(const char *fmt, ...) {
    printf("%s%s[INFO]%s ", ANSI_BOLD, NEON_CYAN, ANSI_RESET);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

void tui_log_success(const char *fmt, ...) {
    printf("%s%s[SUCCESS]%s ", ANSI_BOLD, NEON_GREEN, ANSI_RESET);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

void tui_log_warn(const char *fmt, ...) {
    printf("%s%s[WARN]%s ", ANSI_BOLD, NEON_YELLOW, ANSI_RESET);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

void tui_log_error(const char *fmt, ...) {
    printf("%s%s[ERROR]%s ", ANSI_BOLD, "\033[31m", ANSI_RESET);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════
//  ANIMATIONS & TYPING EFFECTS
// ═══════════════════════════════════════════════════════════════════

void tui_type_text(const char *text, int char_delay_us) {
    if (!text) return;
    if (char_delay_us <= 0) char_delay_us = 10000;

    size_t len = strlen(text);
    for (size_t i = 0; i < len; i++) {
        putchar(text[i]);
        fflush(stdout);
        if (text[i] == ' ' || text[i] == '\n') {
            sleep_us(char_delay_us / 2);
        } else {
            sleep_us(char_delay_us);
        }
    }
}

void tui_show_spinner(const char *message, int duration_ms) {
    const Theme *th = tui_get_theme();
    const char *frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
    int frame_count = sizeof(frames) / sizeof(frames[0]);

    int elapsed = 0;
    int interval_ms = 80;
    int idx = 0;

    printf("\r\033[K");
    while (elapsed < duration_ms) {
        printf("\r%s%s %s %s...%s", ANSI_BOLD, th->primary, frames[idx], message, ANSI_RESET);
        fflush(stdout);
        sleep_us(interval_ms * 1000L);
        elapsed += interval_ms;
        idx = (idx + 1) % frame_count;
    }
    printf("\r\033[K");
    fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════
//  SYNTAX HIGHLIGHTING FOR CODE BLOCKS
// ═══════════════════════════════════════════════════════════════════

static bool is_keyword(const char *word) {
    const char *keywords[] = {
        "int", "char", "float", "double", "void", "return", "if", "else",
        "for", "while", "do", "switch", "case", "break", "continue",
        "struct", "typedef", "static", "const", "import", "def", "class",
        "public", "private", "fn", "let", "mut", NULL
    };
    for (int i = 0; keywords[i] != NULL; i++) {
        if (strcmp(word, keywords[i]) == 0) return true;
    }
    return false;
}

void tui_print_highlighted_code(const char *code_block) {
    if (!code_block) return;
    const Theme *th = tui_get_theme();

    printf("\n%s```%s\n", COLOR_DARK_GRAY, ANSI_RESET);

    const char *p = code_block;
    bool in_string = false;
    char str_quote = '\0';

    while (*p) {
        if (in_string) {
            putchar(*p);
            if (*p == str_quote && *(p - 1) != '\\') {
                in_string = false;
                printf("%s", ANSI_RESET);
            }
            p++;
            continue;
        }

        if (*p == '"' || *p == '\'') {
            in_string = true;
            str_quote = *p;
            printf("%s%c", NEON_GREEN, *p);
            p++;
            continue;
        }

        if (isalpha(*p) || *p == '_') {
            char word[64];
            int idx = 0;
            while ((isalnum(*p) || *p == '_') && idx < 63) {
                word[idx++] = *p++;
            }
            word[idx] = '\0';

            if (is_keyword(word)) {
                printf("%s%s%s%s", ANSI_BOLD, th->primary, word, ANSI_RESET);
            } else {
                printf("%s%s", "\033[37m", word);
            }
            continue;
        }

        if (isdigit(*p)) {
            printf("%s%c%s", NEON_YELLOW, *p, ANSI_RESET);
            p++;
            continue;
        }

        if (*p == '{' || *p == '}' || *p == '(' || *p == ')' || *p == ';' || *p == '=') {
            printf("%s%c%s", th->primary, *p, ANSI_RESET);
            p++;
            continue;
        }

        putchar(*p);
        p++;
    }

    printf("\n%s```%s\n\n", COLOR_DARK_GRAY, ANSI_RESET);
}

// ═══════════════════════════════════════════════════════════════════
//  INTERACTIVE ARROW-KEY NAVIGATION MENU
// ═══════════════════════════════════════════════════════════════════

int tui_interactive_menu(const char *title, const char **options, int option_count) {
    if (option_count <= 0) return -1;
    const Theme *th = tui_get_theme();

    int selected = 0;
    tui_enable_raw_mode();
    printf("\033[?25l");

    while (1) {
        printf("\r\033[K");
        printf("%s%s[%s - Use ↑/↓ Arrow Keys, Enter to Select]%s\n", ANSI_BOLD, th->primary, title ? title : "Menu", ANSI_RESET);

        for (int i = 0; i < option_count; i++) {
            printf("\r\033[K");
            if (i == selected) {
                printf("  %s%s> %-42s%s\n", ANSI_BOLD, th->primary, options[i], ANSI_RESET);
            } else {
                printf("    %s%-42s%s\n", COLOR_DARK_GRAY, options[i], ANSI_RESET);
            }
        }
        fflush(stdout);

        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) break;

        if (c == '\r' || c == '\n') {
            break;
        } else if (c == '\033') {
            char seq[2];
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[0] == '[') {
                    if (seq[1] == 'A') { // UP
                        selected = (selected - 1 + option_count) % option_count;
                    } else if (seq[1] == 'B') { // DOWN
                        selected = (selected + 1) % option_count;
                    }
                }
            }
        } else if (c == 'k' || c == 'K') {
            selected = (selected - 1 + option_count) % option_count;
        } else if (c == 'j' || c == 'J') {
            selected = (selected + 1) % option_count;
        }

        int total_lines = option_count + 1;
        printf("\033[%dA", total_lines);
    }

    printf("\033[?25h");
    tui_disable_raw_mode();
    printf("\n");
    return selected;
}
