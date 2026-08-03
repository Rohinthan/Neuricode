/*
 * tui.h — Expanded Color Theme Options for Neuricode Antigravity CLI
 */

#ifndef NEURICODE_TUI_H
#define NEURICODE_TUI_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

// ═══════════════════════════════════════════════════════════════════
//  THEME PALETTES & COLOR ACCENTS
// ═══════════════════════════════════════════════════════════════════

typedef enum {
    THEME_PURPLE_CLASSIC = 0, // Classic Purple (Default)
    THEME_CYBERPUNK,          // Cyberpunk Cyan / Magenta
    THEME_MATRIX,             // Matrix Electric Green
    THEME_SUNSET_ORANGE,      // Sunset Orange / Gold
    THEME_ELECTRIC_BLUE,      // Cobalt Blue / Cyan
    THEME_CRIMSON_RED,        // Neon Crimson / Coral
    THEME_EMERALD_MINT,       // Emerald Green / Teal
    THEME_MONOCHROME_DARK,    // Slate Gray / Silver Minimal
    THEME_COUNT
} ThemeID;

typedef struct {
    const char *name;
    const char *primary;     // Main accent color
    const char *secondary;   // Secondary accent
    const char *prompt;      // Prompt cursor color
    const char *text;        // Body text
    const char *muted;       // Divider line color
} Theme;

const Theme *tui_get_theme(void);
void tui_set_theme(ThemeID id);
const char *tui_get_theme_name(ThemeID id);

// ANSI Escape Codes
#define ANSI_RESET        "\033[0m"
#define ANSI_BOLD         "\033[1m"
#define ANSI_DIM          "\033[2m"

// Vibrant 24-bit Truecolor Palettes
#define NEON_PURPLE       "\033[38;2;170;90;255m"
#define NEON_MAGENTA      "\033[38;2;220;70;255m"
#define NEON_CYAN         "\033[38;2;0;220;255m"
#define NEON_GREEN        "\033[38;2;50;255;140m"
#define NEON_YELLOW       "\033[38;2;255;210;50m"
#define NEON_ORANGE       "\033[38;2;255;130;40m"
#define NEON_BLUE         "\033[38;2;60;130;255m"
#define NEON_RED          "\033[38;2;255;50;80m"
#define NEON_MINT         "\033[38;2;0;255;160m"
#define COLOR_DARK_GRAY   "\033[38;2;85;90;105m"
#define COLOR_SILVER      "\033[38;2;210;215;225m"

// Terminal & Window Auto-Resize Control
void tui_init(void);
void tui_clear_screen(void);
int  tui_get_terminal_width(void);
void tui_update_terminal_size(void);
void tui_enable_raw_mode(void);
void tui_disable_raw_mode(void);

// Header Banner & Straight Lines
void tui_print_header_banner(const char *version, const char *model_name);

// Straight Input Divider Lines
void tui_print_antigravity_input_frame_start(void);
void tui_print_antigravity_input_frame_end(void);

// Status Badges & Logging
void tui_log_info(const char *fmt, ...);
void tui_log_success(const char *fmt, ...);
void tui_log_warn(const char *fmt, ...);
void tui_log_error(const char *fmt, ...);

// Animations & Effects
void tui_type_text(const char *text, int char_delay_us);
void tui_show_spinner(const char *message, int duration_ms);

// Code Syntax Highlighting
void tui_print_highlighted_code(const char *code_block);

// Interactive Menu
int tui_interactive_menu(const char *title, const char **options, int option_count);

#endif // NEURICODE_TUI_H
