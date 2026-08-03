/*
 * config_ui.c — Configuration Interface Module
 *
 * Terminal User Interface utilizing standard ANSI escape sequences
 * and POSIX termios configuration. Designed for kernel-style menu 
 * management with zero external module dependencies.
 *
 * Copyright (c) 2026. All rights reserved.
 */

#define _POSIX_C_SOURCE 200809L
#include "config_ui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>

/* ── System Hardware Topology Detection ──────────────────────────── */

void detect_system_hardware(HardwareProfile *prof) {
    if (!prof) return;

    /* Query active processing cores via POSIX configuration registry */
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    prof->cpu_cores = (cores > 0) ? (int)cores : 1;

    /* Initialize target buffers with standard system fallbacks */
    strncpy(prof->cpu_model, "Unknown CPU Architecture", sizeof(prof->cpu_model) - 1);
    prof->cpu_model[sizeof(prof->cpu_model) - 1] = '\0';
    prof->gpu_detected = 0;
    strncpy(prof->gpu_name, "None detected", sizeof(prof->gpu_name) - 1);
    prof->gpu_name[sizeof(prof->gpu_name) - 1] = '\0';

    /* Extract exact processor model descriptor via procfs */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    colon++;
                    while (*colon == ' ' || *colon == '\t') colon++;
                    char *nl = strchr(colon, '\n');
                    if (nl) *nl = '\0';
                    strncpy(prof->cpu_model, colon, sizeof(prof->cpu_model) - 1);
                    prof->cpu_model[sizeof(prof->cpu_model) - 1] = '\0';
                    break;
                }
            }
        }
        fclose(f);
    }

    /* Validate graphics subsystem availability via Direct Rendering Manager */
    if (access("/sys/class/drm/renderD128", F_OK) == 0) {
        prof->gpu_detected = 1;
        strncpy(prof->gpu_name, "GPU detected (renderD128)", sizeof(prof->gpu_name) - 1);
    } else if (access("/dev/dri/card0", F_OK) == 0) {
        prof->gpu_detected = 1;
        strncpy(prof->gpu_name, "GPU detected (/dev/dri/card0)", sizeof(prof->gpu_name) - 1);
    }
    prof->gpu_name[sizeof(prof->gpu_name) - 1] = '\0';

    /* Validate NVIDIA/CUDA availability — the neuralc GPU backend is
     * CUDA-only, so this is checked independently of the generic DRM
     * probe above (a DRM device may exist with no CUDA present, e.g.
     * an integrated/AMD GPU, and vice versa on headless CUDA boxes). */
    prof->cuda_detected = 0;
    strncpy(prof->cuda_name, "None detected", sizeof(prof->cuda_name) - 1);
    if (access("/dev/nvidiactl", F_OK) == 0 || access("/dev/nvidia0", F_OK) == 0) {
        prof->cuda_detected = 1;
        strncpy(prof->cuda_name, "NVIDIA driver detected (/dev/nvidia*)",
                sizeof(prof->cuda_name) - 1);
        /* pull the driver version line if present — nice to show the
         * user in the hardware banner without shelling out */
        FILE *nv = fopen("/proc/driver/nvidia/version", "r");
        if (nv) {
            char line[256];
            if (fgets(line, sizeof(line), nv)) {
                char *nl = strchr(line, '\n');
                if (nl) *nl = '\0';
                strncpy(prof->cuda_name, line, sizeof(prof->cuda_name) - 1);
            }
            fclose(nv);
        }
    } else if (access("/usr/lib/x86_64-linux-gnu/libcuda.so", F_OK) == 0 ||
               access("/usr/lib/x86_64-linux-gnu/libcuda.so.1", F_OK) == 0 ||
               access("/usr/local/cuda/lib64/libcudart.so", F_OK) == 0) {
        prof->cuda_detected = 1;
        strncpy(prof->cuda_name, "CUDA runtime library found (no active device node)",
                sizeof(prof->cuda_name) - 1);
    }
    prof->cuda_name[sizeof(prof->cuda_name) - 1] = '\0';
}

/* ── ANSI Escape Sequences & Color Macros ────────────────────────── */
#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define DIM         "\033[2m"

#define FG_BLACK    "\033[30m"
#define FG_RED      "\033[31m"
#define FG_GREEN    "\033[32m"
#define FG_YELLOW   "\033[33m"
#define FG_BLUE     "\033[34m"
#define FG_MAGENTA  "\033[35m"
#define FG_CYAN     "\033[36m"
#define FG_WHITE    "\033[37m"

#define BG_BLACK    "\033[40m"
#define BG_RED      "\033[41m"
#define BG_GREEN    "\033[42m"
#define BG_YELLOW   "\033[43m"
#define BG_BLUE     "\033[44m"
#define BG_MAGENTA  "\033[45m"
#define BG_CYAN     "\033[46m"
#define BG_WHITE    "\033[47m"

#define CLEAR_SCREEN "\033[2J\033[H"
#define CURSOR_HIDE  "\033[?25l"
#define CURSOR_SHOW  "\033[?25h"
#define CURSOR_POS(r,c) printf("\033[%d;%dH", (r), (c))

/* ── POSIX Terminal State Management ──────────────────────────────── */
static struct termios old_term;
static int term_rows = 24;
static int term_cols = 80;

static void term_raw(void) {
    struct termios raw;
    tcgetattr(STDIN_FILENO, &old_term);
    raw = old_term;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    printf(CURSOR_HIDE);
}

static void term_restore(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_term);
    printf(CURSOR_SHOW);
    printf(RESET);
}

static void term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        term_rows = ws.ws_row;
        term_cols = ws.ws_col;
    }
}

/* ── Keyboard Input Parsing ──────────────────────────────────────── */
#define KEY_UP      1000
#define KEY_DOWN    1001
#define KEY_LEFT    1002
#define KEY_RIGHT   1003
#define KEY_ENTER   '\n'
#define KEY_ESC     27
#define KEY_SPACE   ' '

static int read_key(void) {
    unsigned char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return -1;
    if (c == KEY_ESC) {
        unsigned char seq[3];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return KEY_ESC;
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return KEY_ESC;
        if (seq[0] == '[') {
            switch (seq[1]) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
        return KEY_ESC;
    }
    return (int)c;
}

/* ── UI Drawing Primitives ───────────────────────────────────────── */

static void draw_border_box(int row, int col, int h, int w, const char *title) {
    CURSOR_POS(row, col);
    printf(BG_BLACK FG_CYAN "┌");
    if (title && strlen(title) > 0) {
        printf("─ " FG_WHITE BOLD "%s" RESET BG_BLACK FG_CYAN " ", title);
        int used = (int)strlen(title) + 4;
        for (int i = used; i < w - 1; i++) printf("─");
    } else {
        for (int i = 1; i < w - 1; i++) printf("─");
    }
    printf("┐");

    for (int r = 1; r < h - 1; r++) {
        CURSOR_POS(row + r, col);
        printf(BG_BLACK FG_CYAN "│");
        printf(BG_BLACK FG_WHITE);
        for (int i = 1; i < w - 1; i++) printf(" ");
        printf(FG_CYAN "│");
    }

    CURSOR_POS(row + h - 1, col);
    printf(BG_BLACK FG_CYAN "└");
    for (int i = 1; i < w - 1; i++) printf("─");
    printf("┘" RESET);
}

/* ── Element Rendering Engine ────────────────────────────────────── */

static void render_item(const ConfigItem *it, int row, int col, int width, int selected) {
    CURSOR_POS(row, col);

    if (selected)
        printf(BG_WHITE FG_BLACK BOLD);
    else
        printf(BG_BLACK FG_WHITE);

    for (int i = 0; i < width; i++) printf(" ");
    CURSOR_POS(row, col + 1);

    switch (it->type) {
    case ITEM_SEPARATOR:
        printf(BG_BLACK FG_CYAN DIM);
        printf("  ─────────────────────────────────────");
        break;

    case ITEM_INFO:
        if (selected) printf(BG_WHITE FG_BLACK);
        else          printf(BG_BLACK FG_YELLOW);
        printf("  %s", it->label);
        break;

    case ITEM_TOGGLE:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  [%s] %s", it->toggle ? "*" : " ", it->label);
        break;

    case ITEM_RADIO:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  (%s) %s  ", it->radio_opts[it->radio_sel], it->label);
        if (selected) {
            printf(BG_WHITE FG_BLUE);
            printf("< ");
            for (int i = 0; i < it->radio_count; i++) {
                if (i == it->radio_sel)
                    printf(BOLD "(%s)" RESET BG_WHITE FG_BLUE, it->radio_opts[i]);
                else
                    printf(" %s ", it->radio_opts[i]);
                if (i < it->radio_count - 1) printf(" | ");
            }
            printf(" >");
        }
        break;

    case ITEM_NUMBER:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  (%d) %s", it->num_val, it->label);
        if (selected)
            printf(FG_CYAN "  [← → to change, min:%d max:%d]", it->num_min, it->num_max);
        break;

    case ITEM_FLOAT:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_WHITE);
        printf("  (%.4f) %s", it->float_val, it->label);
        if (selected)
            printf(FG_CYAN "  [← to halve, → to double]");
        break;

    case ITEM_SUBMENU:
        if (selected) printf(BG_WHITE FG_BLACK BOLD);
        else          printf(BG_BLACK FG_CYAN);
        printf("  %s  --->", it->label);
        break;
    }

    printf(RESET);
}

/* ── Contextual System Navigation Indicators ────────────────────── */

static void draw_help_bar(int row, const char *help_text) {
    CURSOR_POS(row, 1);
    printf(BG_BLACK FG_CYAN);
    for (int i = 0; i < term_cols; i++) printf("─");

    CURSOR_POS(row + 1, 1);
    printf(BG_BLACK FG_YELLOW);
    printf("  %s", help_text ? help_text : "Use arrows to navigate, Space/Enter to toggle, S to save, Q to quit");
    for (int i = 0; i < term_cols - 2; i++) printf(" ");

    CURSOR_POS(row + 2, 1);
    printf(BG_BLUE FG_WHITE BOLD);
    int bw = term_cols / 5;
    char btns[5][20] = {"<Select>","<Exit>","<Help>","<Save>","<Load>"};
    for (int i = 0; i < 5; i++) {
        printf(BG_WHITE FG_BLACK " %-*s" RESET BG_BLUE, bw-1, btns[i]);
    }
    printf(RESET);
}

/* ── Master Interface Interface Layout ───────────────────────────── */

#define MENU_TOP    4
#define MENU_MARGIN 4

static void draw_menu(Menu *m, const char *breadcrumb) {
    term_size();
    printf(CLEAR_SCREEN);

    CURSOR_POS(1, 1);
    printf(BG_BLUE FG_WHITE BOLD);
    for (int i = 0; i < term_cols; i++) printf(" ");
    CURSOR_POS(1, 1);
    printf("  neuralc Configuration");
    CURSOR_POS(1, term_cols - 20);
    printf("neuralc v0.1  ");
    printf(RESET);

    CURSOR_POS(2, 1);
    printf(BG_BLACK FG_CYAN);
    for (int i = 0; i < term_cols; i++) printf(" ");
    CURSOR_POS(2, 3);
    printf(FG_YELLOW BOLD "► " FG_WHITE "%s", breadcrumb);
    printf(RESET);

    int box_h = term_rows - MENU_TOP - 4;
    int box_w = term_cols - MENU_MARGIN * 2;
    draw_border_box(MENU_TOP, MENU_MARGIN, box_h, box_w, m->title);

    int visible = box_h - 2;
    if (m->cursor < m->scroll) m->scroll = m->cursor;
    if (m->cursor >= m->scroll + visible) m->scroll = m->cursor - visible + 1;
    if (m->scroll < 0) m->scroll = 0;

    for (int i = 0; i < visible && (m->scroll + i) < m->count; i++) {
        int idx = m->scroll + i;
        render_item(&m->items[idx], MENU_TOP + 1 + i, MENU_MARGIN + 1, box_w - 2, idx == m->cursor);
    }

    if (m->count > visible) {
        CURSOR_POS(MENU_TOP + 1, MENU_MARGIN + box_w - 3);
        printf(BG_BLACK FG_CYAN "%d/%d" RESET, m->cursor + 1, m->count);
    }

    const char *help = m->items[m->cursor].help;
    draw_help_bar(term_rows - 2, help[0] ? help : NULL);

    fflush(stdout);
}

/* ── Traversal Index Utilities ───────────────────────────────────── */

static int next_selectable(Menu *m, int from, int dir) {
    int i = from + dir;
    while (i >= 0 && i < m->count) {
        if (m->items[i].type != ITEM_SEPARATOR && m->items[i].type != ITEM_INFO)
            return i;
        i += dir;
    }
    return from;
}

/* ── Interface Modals ────────────────────────────────────────────── */

static void popup_message(const char *title, const char *msg) {
    int w = 50, h = 7;
    int r = (term_rows - h) / 2;
    int c = (term_cols - w) / 2;
    draw_border_box(r, c, h, w, title);
    CURSOR_POS(r + 2, c + 3);
    printf(BG_BLACK FG_WHITE "  %s", msg);
    CURSOR_POS(r + 4, c + w/2 - 4);
    printf(BG_WHITE FG_BLACK BOLD " <  OK  > " RESET);
    fflush(stdout);
    read_key();
}

static void flush_input(void) {
    tcflush(STDIN_FILENO, TCIFLUSH);
}

/* ── Menu Loop Processing Pipeline ────────────────────────────────── */

static int run_menu(Menu *m, const char *breadcrumb);

static int handle_item(Menu *m, int idx, const char *breadcrumb) {
    ConfigItem *it = &m->items[idx];
    switch (it->type) {
    case ITEM_TOGGLE:
        it->toggle = !it->toggle;
        break;
    case ITEM_RADIO:
        it->radio_sel = (it->radio_sel + 1) % it->radio_count;
        break;
    case ITEM_SUBMENU:
        if (it->submenu) {
            char bc[256];
            snprintf(bc, sizeof(bc), "%s > %s", breadcrumb, it->label);
            flush_input();
            int r = run_menu(it->submenu, bc);
            flush_input();
            return r;
        }
        break;
    default:
        break;
    }
    return 0;
}

static int run_menu(Menu *m, const char *breadcrumb) {
    flush_input();
    while (1) {
        draw_menu(m, breadcrumb);
        int k = read_key();
        switch (k) {
        case KEY_UP:
            m->cursor = next_selectable(m, m->cursor, -1);
            break;
        case KEY_DOWN:
            m->cursor = next_selectable(m, m->cursor, +1);
            break;
        case KEY_LEFT: {
            ConfigItem *it = &m->items[m->cursor];
            if (it->type == ITEM_NUMBER && it->num_val > it->num_min)
                it->num_val--;
            else if (it->type == ITEM_FLOAT)
                it->float_val /= 2.0f;
            else if (it->type == ITEM_RADIO)
                it->radio_sel = (it->radio_sel + it->radio_count - 1) % it->radio_count;
            break;
        }
        case KEY_RIGHT: {
            ConfigItem *it = &m->items[m->cursor];
            if (it->type == ITEM_NUMBER && it->num_val < it->num_max)
                it->num_val++;
            else if (it->type == ITEM_FLOAT)
                it->float_val *= 2.0f;
            else if (it->type == ITEM_RADIO)
                it->radio_sel = (it->radio_sel + 1) % it->radio_count;
            break;
        }
        case KEY_ENTER:
        case KEY_SPACE:
            if (handle_item(m, m->cursor, breadcrumb) == 1)
                return 1;
            break;
        case 'y': case 'Y':
            if (m->items[m->cursor].type == ITEM_TOGGLE)
                m->items[m->cursor].toggle = 1;
            break;
        case 'n': case 'N':
            if (m->items[m->cursor].type == ITEM_TOGGLE)
                m->items[m->cursor].toggle = 0;
            break;
        case 's': case 'S':
            return 1;
        case 'q': case 'Q':
        case KEY_ESC:
            return 0;
        case 'h': case 'H': case '?':
            popup_message("Help", "↑↓ Navigate  Space/Enter Toggle  ←→ Change value  S Save  Q Quit");
            break;
        }
    }
}

/* ── Menu Initialization Helpers ──────────────────────────────────── */

static void add_sep(Menu *m, const char *label) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type = ITEM_SEPARATOR;
    snprintf(it->label, MAX_LABEL, "%s", label ? label : "");
}

static void add_toggle(Menu *m, const char *label, const char *key, int val, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type   = ITEM_TOGGLE;
    it->toggle = val;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_radio(Menu *m, const char *label, const char *key, int sel, int n, const char **opts, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type      = ITEM_RADIO;
    it->radio_sel = sel;
    it->radio_count = n;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    for (int i = 0; i < n && i < 8; i++)
        snprintf(it->radio_opts[i], 32, "%s", opts[i]);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_number(Menu *m, const char *label, const char *key, int val, int mn, int mx, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type    = ITEM_NUMBER;
    it->num_val = val;
    it->num_min = mn;
    it->num_max = mx;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_float(Menu *m, const char *label, const char *key, float val, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type      = ITEM_FLOAT;
    it->float_val = val;
    snprintf(it->label, MAX_LABEL, "%s", label);
    snprintf(it->key,   64,        "%s", key);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_submenu(Menu *m, const char *label, Menu *sub, const char *help) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type    = ITEM_SUBMENU;
    it->submenu = sub;
    snprintf(it->label, MAX_LABEL, "%s", label);
    if (help) snprintf(it->help, 256, "%s", help);
}

static void add_info(Menu *m, const char *text) {
    ConfigItem *it = &m->items[m->count++];
    memset(it, 0, sizeof(*it));
    it->type = ITEM_INFO;
    snprintf(it->label, MAX_LABEL, "%s", text);
}

/* ── Menu Hierarchical Model Tree Specification ───────────────────── */

static Menu perf_menu, train_menu, mem_menu, debug_menu, build_menu, trans_menu;
static Menu root_menu;

static void build_menus(NeuralcConfig *cfg) {
    memset(&perf_menu,  0, sizeof(Menu));
    memset(&train_menu, 0, sizeof(Menu));
    memset(&mem_menu,   0, sizeof(Menu));
    memset(&debug_menu, 0, sizeof(Menu));
    memset(&build_menu, 0, sizeof(Menu));
    memset(&trans_menu, 0, sizeof(Menu));
    memset(&root_menu,  0, sizeof(Menu));

    /* ── Performance Submenu ── */
    snprintf(perf_menu.title, MAX_LABEL, "Performance Settings");
    add_info(&perf_menu, "Configure CPU/GPU acceleration options");
    add_sep(&perf_menu, NULL);
    add_toggle(&perf_menu, "Enable OpenMP multi-core", "USE_OMP", cfg->use_omp, "Use multiple CPU cores in parallel for tensor operations");
    {
        const char *ta[] = {"Auto", "Manual"};
        add_radio(&perf_menu, "Thread Allocation", "OMP_THREAD_MODE", cfg->omp_auto ? 0 : 1, 2, ta, "Auto = detect all CPU cores, Manual = set count below");
    }
    add_number(&perf_menu, "Thread Count (Manual mode)", "OMP_THREADS", cfg->omp_threads, 1, 256, "Number of threads for OpenMP (only used in Manual mode)");
    add_sep(&perf_menu, NULL);
    add_toggle(&perf_menu, "Enable CUDA GPU acceleration", "USE_GPU", cfg->use_gpu, "Offload matmul/Conv2D/MaxPool2D/activations to an NVIDIA GPU via CUDA (requires nvcc + a CUDA-capable device)");
    add_toggle(&perf_menu, "Enable BLAS integration", "USE_BLAS", cfg->use_blas, "Use OpenBLAS for faster matrix multiply");

    /* ── Training Submenu ── */
    snprintf(train_menu.title, MAX_LABEL, "Training & Model Defaults");
    add_info(&train_menu, "Default hyperparameters for training & inference");
    add_sep(&train_menu, NULL);
    add_number(&train_menu, "Hidden Layer Size", "HIDDEN_SIZE", cfg->hidden_size, 16, 4096, "Hidden state size for RNN/Dense layers (e.g., 128, 256, 512)");
    add_number(&train_menu, "Sequence Length (seq_len)", "SEQ_LEN", cfg->seq_len, 1, 1024, "Sequence length per batch step during training");
    add_float(&train_menu, "Sampling Temperature", "TEMPERATURE", cfg->temperature, "Sampling temperature for logit scaling during generation (e.g. 0.7 - 1.0)");
    add_sep(&train_menu, NULL);
    add_number(&train_menu, "Default Batch Size", "DEFAULT_BATCH", cfg->batch_size, 1, 4096, "Samples per gradient update. Larger = faster but more memory");
    add_float(&train_menu, "Default Learning Rate", "DEFAULT_LR", cfg->learning_rate, "Step size for gradient descent. Adam typical: 0.001, SGD: 0.01");
    add_number(&train_menu, "Epoch Cycles", "DEFAULT_EPOCHS", cfg->epochs, 1, 10000, "Number of full passes over the training dataset");
    add_sep(&train_menu, NULL);
    {
        const char *opts[] = {"Adam","SGD","RMSProp"};
        add_radio(&train_menu, "Default Optimizer", "DEFAULT_OPT", cfg->optimizer, 3, opts, "Adam is best for most tasks. SGD needs tuned LR. RMSProp good for RNN");
    }
    add_float(&train_menu, "Default Dropout Rate", "DEFAULT_DROPOUT", cfg->dropout_rate, "Fraction of neurons to drop during training. 0.0 = disabled");
    add_sep(&train_menu, NULL);
    add_toggle(&train_menu, "Enable Gradient Clipping", "USE_GRAD_CLIP", cfg->use_grad_clip, "Prevent exploding gradients — essential for RNN/LSTM");
    add_float(&train_menu, "Gradient Clip Norm", "GRAD_CLIP_NORM", cfg->grad_clip, "Maximum gradient L2 norm. Typical: 1.0 for RNN, 5.0 for Dense");

    /* ── Transformer Engine Submenu ── */
    snprintf(trans_menu.title, MAX_LABEL, "Transformer Engine Hyperparameters");
    add_info(&trans_menu, "Configure Multi-Head Attention & Decoder Transformer parameters");
    add_sep(&trans_menu, NULL);
    add_number(&trans_menu, "Vocabulary Size (TRANS_VOCAB_SIZE)", "TRANS_VOCAB_SIZE", cfg->trans_vocab_size, 100, 128000, "BPE / Tokenizer vocabulary size (default: 32000)");
    add_number(&trans_menu, "Hidden Dimension (TRANS_DIM)",        "TRANS_DIM",        cfg->trans_dim,        64, 8192,   "Model embedding dimension d_model (default: 768)");
    add_number(&trans_menu, "FFN Expansion Dim (TRANS_HIDDEN_DIM)","TRANS_HIDDEN_DIM", cfg->trans_hidden_dim, 128, 32768, "SwiGLU FFN interior dimension (default: 2048)");
    add_number(&trans_menu, "Number of Layers (TRANS_N_LAYERS)",  "TRANS_N_LAYERS",   cfg->trans_n_layers,     1, 128,    "Number of stacked Attention + FFN blocks (default: 12)");
    add_number(&trans_menu, "Number of Heads (TRANS_N_HEADS)",    "TRANS_N_HEADS",    cfg->trans_n_heads,      1, 64,     "Number of multi-head attention heads (default: 12)");
    add_number(&trans_menu, "Max Context Length (TRANS_MAX_SEQ_LEN)","TRANS_MAX_SEQ_LEN",cfg->trans_max_seq_len, 64, 131072, "Maximum sequence KV-Cache length (default: 2048)");

    /* ── Memory Submenu ── */
    snprintf(mem_menu.title, MAX_LABEL, "Memory Settings");
    add_info(&mem_menu, "Control memory allocation strategy");
    add_sep(&mem_menu, NULL);
    {
        const char *ma[] = {"malloc","Pool"};
        add_radio(&mem_menu, "Memory Allocator", "ALLOCATOR", cfg->allocator, 2, ma, "malloc = standard, Pool = pre-allocate for speed");
    }
    add_number(&mem_menu, "Memory Pool Size (MB)", "POOL_SIZE_MB", cfg->pool_size_mb, 64, 16384, "Size of pre-allocated memory pool (Pool allocator only)");

    /* ── Debug Submenu ── */
    snprintf(debug_menu.title, MAX_LABEL, "Debug & Profiling");
    add_info(&debug_menu, "Enable diagnostic and profiling features");
    add_sep(&debug_menu, NULL);
    add_toggle(&debug_menu, "Debug Mode (verbose logging)", "DEBUG_MODE", cfg->debug_mode, "Print detailed logs during training. Slows down execution");
    add_toggle(&debug_menu, "Check for NaN in tensors", "CHECK_NAN", cfg->check_nan, "Detect NaN/Inf values in tensors. Useful for debugging instability");
    add_toggle(&debug_menu, "Profiling (print timing)", "PROFILE", cfg->profile, "Print timing information for each operation");

    /* ── Build Submenu ── */
    snprintf(build_menu.title, MAX_LABEL, "Build Options");
    add_info(&build_menu, "Compiler and build configuration");
    add_sep(&build_menu, NULL);
    {
        const char *ol[] = {"-O0","-O1","-O2","-O3"};
        add_radio(&build_menu, "Optimization Level", "OPT_LEVEL", cfg->opt_level, 4, ol, "-O2 is default. -O3 is faster but may cause issues. -O0 for debugging");
    }
    add_toggle(&build_menu, "Enable AVX/SIMD instructions", "ENABLE_AVX", cfg->enable_avx, "Use CPU vector instructions for faster math. Requires modern CPU");
    add_toggle(&build_menu, "Enable Link-Time Optimization", "ENABLE_LTO", cfg->enable_lto, "Optimize across compilation units. Slower build, faster runtime");

    /* ── Root Menu Configuration ── */
    snprintf(root_menu.title, MAX_LABEL, "neuralc v0.1 Configuration");
    add_info(&root_menu, "Arrow keys navigate. Space/Enter toggle. S save. Q quit.");
    add_sep(&root_menu, NULL);
    {
        const char *archs[] = {"RNN", "Transformer Engine"};
        add_radio(&root_menu, "Model Architecture Selection", "ACTIVE_MODEL_TYPE", cfg->active_model_type, 2, archs, "Select between RNN and Decoder Transformer with KV-Cache");
    }
    add_submenu(&root_menu, "Performance Settings",   &perf_menu, "OpenMP threads, GPU, BLAS");
    add_submenu(&root_menu, "Training Defaults",      &train_menu, "Hidden size, Temp, Batch size, LR, Epochs");
    add_submenu(&root_menu, "Transformer Engine Setup",&trans_menu, "Vocabulary size, Dim, Layers, Heads, Context Length");
    add_submenu(&root_menu, "Memory Settings",        &mem_menu, "Allocator and pool size");
    add_submenu(&root_menu, "Debug & Profiling",      &debug_menu, "Logging, NaN check, timing");
    add_submenu(&root_menu, "Build Options",          &build_menu, "Compiler flags, AVX, LTO");
    add_sep(&root_menu, NULL);
    add_info(&root_menu, "Press S to save  |  Q to quit without saving");
}

/* ── UI Struct Field Aggregation Pipeline ────────────────────────── */

static void collect_config(NeuralcConfig *cfg) {
    /* root menu model selection */
    cfg->active_model_type = root_menu.items[2].radio_sel;

    /* performance */
    cfg->use_omp      = perf_menu.items[2].toggle;
    cfg->omp_auto     = (perf_menu.items[3].radio_sel == 0);
    cfg->omp_threads  = perf_menu.items[4].num_val;
    cfg->use_gpu      = perf_menu.items[6].toggle;
    cfg->use_blas     = perf_menu.items[7].toggle;

    /* training */
    cfg->hidden_size  = train_menu.items[2].num_val;
    cfg->seq_len      = train_menu.items[3].num_val;
    cfg->temperature  = train_menu.items[4].float_val;
    cfg->batch_size   = train_menu.items[6].num_val;
    cfg->learning_rate= train_menu.items[7].float_val;
    cfg->epochs       = train_menu.items[8].num_val;
    cfg->optimizer    = train_menu.items[10].radio_sel;
    cfg->dropout_rate = train_menu.items[11].float_val;
    cfg->use_grad_clip= train_menu.items[13].toggle;
    cfg->grad_clip    = train_menu.items[14].float_val;

    /* transformer */
    cfg->trans_vocab_size  = trans_menu.items[2].num_val;
    cfg->trans_dim         = trans_menu.items[3].num_val;
    cfg->trans_hidden_dim  = trans_menu.items[4].num_val;
    cfg->trans_n_layers    = trans_menu.items[5].num_val;
    cfg->trans_n_heads     = trans_menu.items[6].num_val;
    cfg->trans_max_seq_len = trans_menu.items[7].num_val;

    /* memory */
    cfg->allocator    = mem_menu.items[2].radio_sel;
    cfg->pool_size_mb = mem_menu.items[3].num_val;

    /* debug */
    cfg->debug_mode   = debug_menu.items[2].toggle;
    cfg->check_nan    = debug_menu.items[3].toggle;
    cfg->profile      = debug_menu.items[4].toggle;

    /* build */
    cfg->opt_level    = build_menu.items[2].radio_sel;
    cfg->enable_avx   = build_menu.items[3].toggle;
    cfg->enable_lto   = build_menu.items[4].toggle;
}

/* ── Public API Core Interfaces ──────────────────────────────────── */

void config_defaults(NeuralcConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->active_model_type = 1;
    cfg->use_omp       = 1;
    cfg->omp_auto      = 1;
    cfg->omp_threads   = 4;
    cfg->use_gpu       = 0;
    cfg->use_blas      = 0;
    cfg->hidden_size   = 256;
    cfg->seq_len       = 20;
    cfg->temperature   = 0.8f;
    cfg->batch_size    = 8;
    cfg->learning_rate = 0.01f;
    cfg->optimizer     = 0;
    cfg->dropout_rate  = 0.3f;
    cfg->epochs        = 20;
    cfg->grad_clip     = 1.0f;
    cfg->use_grad_clip = 1;
    cfg->trans_vocab_size  = 32000;
    cfg->trans_dim         = 768;
    cfg->trans_hidden_dim  = 2048;
    cfg->trans_n_layers    = 12;
    cfg->trans_n_heads     = 12;
    cfg->trans_max_seq_len = 2048;
    cfg->allocator     = 0;
    cfg->pool_size_mb  = 512;
    cfg->debug_mode    = 0;
    cfg->check_nan     = 0;
    cfg->profile       = 0;
    cfg->opt_level     = 2;
    cfg->enable_avx    = 0;
    cfg->enable_lto    = 0;
}

int config_ui_run(NeuralcConfig *cfg) {
    HardwareProfile hw;
    detect_system_hardware(&hw);

    if (cfg->omp_auto && cfg->omp_threads < 1)
        cfg->omp_threads = hw.cpu_cores;

    term_raw();
    term_size();
    build_menus(cfg);

    printf(CLEAR_SCREEN);
    printf(BG_BLUE FG_WHITE BOLD);
    printf("  System Hardware Diagnostic Topology\n" RESET);
    printf(BG_BLACK FG_CYAN);
    printf("  CPU Model: " FG_WHITE "%s\n" RESET, hw.cpu_model);
    printf(BG_BLACK FG_CYAN);
    printf("  Logical Processors: " FG_GREEN "%d" FG_CYAN "\n" RESET, hw.cpu_cores);
    printf(BG_BLACK FG_CYAN);
    printf("  GPU Hardware Accel: %s\n" RESET, hw.gpu_detected ? FG_GREEN "Detected" : FG_YELLOW "Not detected");
    printf(BG_BLACK FG_CYAN);
    printf("  CUDA (NVIDIA):      %s" FG_WHITE " %s\n" RESET,
           hw.cuda_detected ? FG_GREEN "Detected   " : FG_YELLOW "Not detected",
           hw.cuda_detected ? hw.cuda_name : "");
    printf(RESET "\n  Loading localized matrix parameters...\n\n" RESET);
    fflush(stdout);

    struct timespec ts = {0, 600000000L};
    nanosleep(&ts, NULL);

    root_menu.cursor = next_selectable(&root_menu, -1, +1);
    int saved = run_menu(&root_menu, "Main Menu");

    printf(CLEAR_SCREEN);
    term_restore();

    if (!saved) {
        printf("\nConfiguration aborted.\n");
        return 1;
    }

    collect_config(cfg);

    printf("\n" BOLD "Configuration Successfully Preserved\n" RESET);
    printf("─────────────────────────────────────────\n");

    if (cfg->use_omp) {
        int threads = cfg->omp_auto ? 4 : cfg->omp_threads;
        printf("  OpenMP Core Binding:  " FG_GREEN "Active" RESET "\n");
        printf("  Thread Safe Context:  ");
        fflush(stdout);
        for (int t = 1; t <= threads; t++) {
            printf(FG_CYAN BOLD "[%d]" RESET " ", t);
            fflush(stdout);
            struct timespec ts = {0, 100000000L};
            nanosleep(&ts, NULL);
        }
        if (cfg->omp_auto)
            printf(FG_GREEN "✓ Auto allocation mode\n" RESET);
        else
            printf(FG_GREEN "✓ %d thread context lock\n" RESET, threads);
    } else {
        printf("  OpenMP Core Binding:  " FG_YELLOW "Inactive (Sequential Engine Allocation)\n" RESET);
    }

    const char *ol[]  = {"-O0","-O1","-O2","-O3"};
    const char *opt[] = {"Adam","SGD","RMSProp"};
    if (cfg->use_gpu && !hw.cuda_detected) {
        printf("  GPU Pipeline target:  " FG_YELLOW "Enabled in config, but no CUDA device "
               "was detected on this machine\n" RESET);
        printf("                         Build with -DUSE_CUDA anyway if you like, but guard\n"
               "                         tensor_to_gpu() calls with cf_cuda_enabled() so\n"
               "                         training falls back to CPU instead of aborting\n");
    } else {
        printf("  GPU Pipeline target:  %s\n", cfg->use_gpu ? FG_GREEN "Enabled (CUDA)" RESET : "Disabled");
    }
    printf("  Model Architecture:   %s\n", cfg->active_model_type == 1 ? FG_GREEN "Transformer Engine" RESET : "RNN");
    printf("  Optimization Profile: %s\n", ol[cfg->opt_level < 4 ? cfg->opt_level : 2]);
    printf("  Hidden Layer Size:    %d\n",   cfg->hidden_size);
    printf("  Sequence Length:      %d\n",   cfg->seq_len);
    printf("  Sampling Temperature: %.4f\n", cfg->temperature);
    printf("  Matrix Batch Boundary: %d\n",   cfg->batch_size);
    printf("  Core Learning Rate:   %.6f\n", cfg->learning_rate);
    printf("  Epoch Cycles:         %d\n",   cfg->epochs);
    printf("  Mathematical Optimizer: %s\n",   opt[cfg->optimizer < 3 ? cfg->optimizer : 0]);
    printf("  Gradient Clip Bounds:  %s (L2 norm limit = %.1f)\n", cfg->use_grad_clip ? "Active" : "Inactive", cfg->grad_clip);
    printf("  Verbose Logging Trace: %s\n",   cfg->debug_mode ? FG_YELLOW "Active" RESET : "Inactive");
    printf("─────────────────────────────────────────\n");
    printf(FG_GREEN BOLD "\n✓ Parameter manifest written successfully\n\n" RESET);
    return 0;
}

int config_save(const NeuralcConfig *cfg, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "config_save: cannot write target file path destination '%s'\n", path);
        return -1;
    }

    const char *opt_names[] = {"-O0", "-O1", "-O2", "-O3"};
    const char *opt_names2[] = {"ADAM", "SGD", "RMSPROP"};

    fprintf(f,
"/*\n"
" * neuralc_config.h — System Config Manifest\n"
" * Auto-generated parameters block — modifications will be overwritten\n"
" */\n"
"#ifndef NEURALC_CONFIG_H\n"
"#define NEURALC_CONFIG_H\n\n"
"#define NEURALC_HAS_CONFIG       1\n\n"
"/* ── Model Architecture Selection ───────────────────── */\n"
"#define MODEL_TYPE_RNN          0\n"
"#define MODEL_TYPE_TRANSFORMER   1\n"
"#define ACTIVE_MODEL_TYPE       %d\n\n"
"/* ── Transformer Hyperparameters ─────────────────────── */\n"
"#define TRANS_VOCAB_SIZE        %d\n"
"#define TRANS_DIM               %d\n"
"#define TRANS_HIDDEN_DIM        %d\n"
"#define TRANS_N_LAYERS          %d\n"
"#define TRANS_N_HEADS           %d\n"
"#define TRANS_MAX_SEQ_LEN       %d\n\n"
"/* ── Performance ────────────────────────────────────── */\n"
"#define NEURALC_USE_OMP         %d\n"
"#define NEURALC_OMP_AUTO        %d\n"
"#define NEURALC_OMP_THREADS     %d\n"
"#define NEURALC_USE_GPU         %d\n"
"#define NEURALC_USE_BLAS        %d\n\n"
"/* ── Training & Model Hyperparameters ────────────────── */\n"
"#define HIDDEN_SIZE             %d\n"
"#define SEQ_LEN                 %d\n"
"#define TEMPERATURE             %.4ff\n"
"#define BATCH_SIZE              %d\n"
"#define LEARNING_RATE           %.6ff\n"
"#define EPOCHS                  %d\n\n"
"#define NEURALC_HIDDEN_SIZE     %d\n"
"#define NEURALC_SEQ_LEN         %d\n"
"#define NEURALC_TEMPERATURE     %.4ff\n"
"#define NEURALC_BATCH_SIZE      %d\n"
"#define NEURALC_LR              %.6ff\n"
"#define NEURALC_OPTIMIZER       %s\n"
"#define NEURALC_DROPOUT         %.4ff\n"
"#define NEURALC_EPOCHS          %d\n"
"#define NEURALC_GRAD_CLIP       %.4ff\n"
"#define NEURALC_USE_GRAD_CLIP   %d\n\n"
"/* ── Memory ─────────────────────────────────────────── */\n"
"#define NEURALC_ALLOCATOR       %d\n"
"#define NEURALC_POOL_MB         %d\n\n"
"/* ── Debug ──────────────────────────────────────────── */\n"
"#define NEURALC_DEBUG           %d\n"
"#define NEURALC_CHECK_NAN       %d\n"
"#define NEURALC_PROFILE         %d\n\n"
"/* ── Build ──────────────────────────────────────────── */\n"
"#define NEURALC_OPT_LEVEL       \"%s\"\n"
"#define NEURALC_AVX             %d\n"
"#define NEURALC_LTO             %d\n\n"
"/* ── Compilation Environment Injections ─────────────── */\n"
"#if NEURALC_USE_OMP\n"
"  #ifndef USE_OMP\n"
"    #define USE_OMP\n"
"  #endif\n"
"#endif\n\n"
"#if NEURALC_USE_GPU\n"
"  #ifndef USE_CUDA\n"
"    #define USE_CUDA\n"
"  #endif\n"
"#endif\n\n"
"#endif\n",
        cfg->active_model_type,
        cfg->trans_vocab_size,
        cfg->trans_dim,
        cfg->trans_hidden_dim,
        cfg->trans_n_layers,
        cfg->trans_n_heads,
        cfg->trans_max_seq_len,
        cfg->use_omp,
        cfg->omp_auto,
        cfg->omp_threads,
        cfg->use_gpu,
        cfg->use_blas,
        cfg->hidden_size,
        cfg->seq_len,
        cfg->temperature,
        cfg->batch_size,
        cfg->learning_rate,
        cfg->epochs,
        cfg->hidden_size,
        cfg->seq_len,
        cfg->temperature,
        cfg->batch_size,
        cfg->learning_rate,
        opt_names2[cfg->optimizer < 3 ? cfg->optimizer : 0],
        cfg->dropout_rate,
        cfg->epochs,
        cfg->grad_clip,
        cfg->use_grad_clip,
        cfg->allocator,
        cfg->pool_size_mb,
        cfg->debug_mode,
        cfg->check_nan,
        cfg->profile,
        opt_names[cfg->opt_level < 4 ? cfg->opt_level : 2],
        cfg->enable_avx,
        cfg->enable_lto
    );

    fclose(f);
    return 0;
}

int config_load(NeuralcConfig *cfg, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    config_defaults(cfg);

    while (fgets(line, sizeof(line), f)) {
        int iv; float fv;
        char sv[64];
        if (sscanf(line, "#define ACTIVE_MODEL_TYPE %d",  &iv)==1) cfg->active_model_type=iv;
        if (sscanf(line, "#define TRANS_VOCAB_SIZE %d",   &iv)==1) cfg->trans_vocab_size=iv;
        if (sscanf(line, "#define TRANS_DIM %d",          &iv)==1) cfg->trans_dim=iv;
        if (sscanf(line, "#define TRANS_HIDDEN_DIM %d",   &iv)==1) cfg->trans_hidden_dim=iv;
        if (sscanf(line, "#define TRANS_N_LAYERS %d",     &iv)==1) cfg->trans_n_layers=iv;
        if (sscanf(line, "#define TRANS_N_HEADS %d",      &iv)==1) cfg->trans_n_heads=iv;
        if (sscanf(line, "#define TRANS_MAX_SEQ_LEN %d",  &iv)==1) cfg->trans_max_seq_len=iv;
        if (sscanf(line, "#define NEURALC_USE_OMP %d",     &iv)==1) cfg->use_omp=iv;
        if (sscanf(line, "#define NEURALC_OMP_AUTO %d",    &iv)==1) cfg->omp_auto=iv;
        if (sscanf(line, "#define NEURALC_OMP_THREADS %d", &iv)==1) cfg->omp_threads=iv;
        if (sscanf(line, "#define NEURALC_USE_GPU %d",     &iv)==1) cfg->use_gpu=iv;
        if (sscanf(line, "#define NEURALC_USE_BLAS %d",    &iv)==1) cfg->use_blas=iv;
        if (sscanf(line, "#define HIDDEN_SIZE %d",         &iv)==1) cfg->hidden_size=iv;
        if (sscanf(line, "#define SEQ_LEN %d",             &iv)==1) cfg->seq_len=iv;
        if (sscanf(line, "#define TEMPERATURE %ff",        &fv)==1) cfg->temperature=fv;
        if (sscanf(line, "#define BATCH_SIZE %d",          &iv)==1) cfg->batch_size=iv;
        if (sscanf(line, "#define LEARNING_RATE %ff",      &fv)==1) cfg->learning_rate=fv;
        if (sscanf(line, "#define EPOCHS %d",              &iv)==1) cfg->epochs=iv;
        if (sscanf(line, "#define NEURALC_HIDDEN_SIZE %d", &iv)==1) cfg->hidden_size=iv;
        if (sscanf(line, "#define NEURALC_SEQ_LEN %d",     &iv)==1) cfg->seq_len=iv;
        if (sscanf(line, "#define NEURALC_TEMPERATURE %ff",&fv)==1) cfg->temperature=fv;
        if (sscanf(line, "#define NEURALC_BATCH_SIZE %d",  &iv)==1) cfg->batch_size=iv;
        if (sscanf(line, "#define NEURALC_LR %ff",         &fv)==1) cfg->learning_rate=fv;
        
        if (sscanf(line, "#define NEURALC_OPTIMIZER %63s", sv)==1) {
            if (strcmp(sv, "ADAM") == 0) cfg->optimizer = 0;
            else if (strcmp(sv, "SGD") == 0) cfg->optimizer = 1;
            else if (strcmp(sv, "RMSPROP") == 0) cfg->optimizer = 2;
        }

        if (sscanf(line, "#define NEURALC_DROPOUT %ff",    &fv)==1) cfg->dropout_rate=fv;
        if (sscanf(line, "#define NEURALC_EPOCHS %d",      &iv)==1) cfg->epochs=iv;
        if (sscanf(line, "#define NEURALC_GRAD_CLIP %ff",  &fv)==1) cfg->grad_clip=fv;
        if (sscanf(line, "#define NEURALC_USE_GRAD_CLIP %d",&iv)==1) cfg->use_grad_clip=iv;
        if (sscanf(line, "#define NEURALC_ALLOCATOR %d",   &iv)==1) cfg->allocator=iv;
        if (sscanf(line, "#define NEURALC_POOL_MB %d",     &iv)==1) cfg->pool_size_mb=iv;
        if (sscanf(line, "#define NEURALC_DEBUG %d",       &iv)==1) cfg->debug_mode=iv;
        if (sscanf(line, "#define NEURALC_CHECK_NAN %d",   &iv)==1) cfg->check_nan=iv;
        if (sscanf(line, "#define NEURALC_PROFILE %d",     &iv)==1) cfg->profile=iv;

        if (sscanf(line, "#define NEURALC_OPT_LEVEL \"%63[^\"]\"", sv)==1) {
            if (strcmp(sv, "-O0") == 0) cfg->opt_level = 0;
            else if (strcmp(sv, "-O1") == 0) cfg->opt_level = 1;
            else if (strcmp(sv, "-O2") == 0) cfg->opt_level = 2;
            else if (strcmp(sv, "-O3") == 0) cfg->opt_level = 3;
        }

        if (sscanf(line, "#define NEURALC_AVX %d",         &iv)==1) cfg->enable_avx=iv;
        if (sscanf(line, "#define NEURALC_LTO %d",         &iv)==1) cfg->enable_lto=iv;
    }
    fclose(f);
    return 0;
}

void config_print(const NeuralcConfig *cfg) {
    const char *opts[] = {"Adam","SGD","RMSProp"};
    printf("\n=== Engine Parameters Configuration Map ===\n");
    printf("Execution Engine Profile:\n");
    printf("  Multi-Core Cluster Allocation (OpenMP): %s", cfg->use_omp ? "Active" : "Inactive");
    if (cfg->use_omp)
        printf(" (%s mode, active threads: %d)", cfg->omp_auto ? "Auto" : "Manual", cfg->omp_threads);
    printf("\n");
    printf("  Graphics Pipeline Engine Core (CUDA): %s\n", cfg->use_gpu  ? "Active" : "Inactive");
    printf("  BLAS Hardware Vectorization Engine:    %s\n", cfg->use_blas ? "Active" : "Inactive");
    printf("Hyperparameters Layout:\n");
    printf("  Matrix Batch Limit Segment:            %d\n", cfg->batch_size);
    printf("  Global Optimization Step Coefficient:  %.6f\n", cfg->learning_rate);
    printf("  Mathematical Optimization Algorithm:   %s\n", opts[cfg->optimizer < 3 ? cfg->optimizer : 0]);
    printf("  Dropout Layer Regularization Threshold: %.2f\n", cfg->dropout_rate);
    printf("  Training Iteration Boundaries (Epochs): %d\n",   cfg->epochs);
    printf("  Gradient Norm Explosion Clipping:      %s (Threshold bound=%.2f)\n", cfg->use_grad_clip ? "Active" : "Inactive", cfg->grad_clip);
    printf("Diagnostics:\n");
    printf("  Verbose Engine Logging Stream Trace:   %s\n", cfg->debug_mode ? "Active" : "Inactive");
    printf("  NaN/Inf Arithmetic Protection Mask:    %s\n", cfg->check_nan  ? "Active" : "Inactive");
    printf("  Execution Timing Profiler Runtime:     %s\n", cfg->profile    ? "Active" : "Inactive");
    printf("Compilation Flags:\n");
    const char *ol[] = {"-O0","-O1","-O2","-O3"};
    printf("  Binary Optimization Direct Profile:    %s\n", ol[cfg->opt_level < 4 ? cfg->opt_level : 2]);
    printf("  Advanced Vector Extensions (AVX-2/SIMD): %s\n", cfg->enable_avx ? "Active" : "Inactive");
    printf("  Link-Time Cross-Module Optimization (LTO): %s\n", cfg->enable_lto ? "Active" : "Inactive");
    printf("===========================================\n\n");
}
