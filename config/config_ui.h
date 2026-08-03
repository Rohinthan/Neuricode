#ifndef CONFIG_UI_H
#define CONFIG_UI_H

/*
 *
 * config_ui.h — neuralc kernel-style configuration UI
 *
 * Terminal UI similar to Linux menuconfig.
 * Uses ANSI escape codes + termios — zero external dependencies.
 *
 * Usage:
 *   make config     (builds + auto-runs menuconfig)
 *
 * Generates: neuralc_config.h
 *
 * Navigation:
 *   ↑ ↓    — move between options
 *   Enter  — select / enter submenu
 *   Space  — toggle checkbox
 *   Y / N  — enable / disable
 *   S      — save config
 *   Q/Esc  — back / quit
 *   H      — help
 */

#include <stddef.h>

/* ── Hardware Profile ────────────────────── */
/*
 * Detected at startup by detect_system_hardware().
 * Used to set smart defaults in the config UI
 * (e.g. auto-fill thread count with actual CPU cores).
 */
typedef struct {
    int  cpu_cores;        /* logical CPU cores via sysconf()    */
    char cpu_model[128];   /* from /proc/cpuinfo "model name"    */
    int  gpu_detected;     /* 1 if DRM/DRI device found          */
    char gpu_name[128];    /* GPU device path or name            */
    int  cuda_detected;    /* 1 if an NVIDIA/CUDA device is found */
    char cuda_name[128];   /* CUDA device path or driver info     */
} HardwareProfile;

/* ── item types ─────────────────────────────────────────────────── */
typedef enum {
    ITEM_TOGGLE,    /* [*] / [ ]  on/off checkbox       */
    ITEM_RADIO,     /* (*) / ( )  one of N choices      */
    ITEM_NUMBER,    /* integer value                     */
    ITEM_FLOAT,     /* float value                       */
    ITEM_SUBMENU,   /* ---> opens child menu            */
    ITEM_SEPARATOR, /* ─────── visual divider           */
    ITEM_INFO,      /* read-only info line               */
} ItemType;

#define MAX_ITEMS   64
#define MAX_LABEL   80

/* ── single config item ─────────────────────────────────────────── */
typedef struct ConfigItem {
    ItemType    type;
    char        label[MAX_LABEL];
    char        key[64];
    char        help[256];
    int         toggle;
    int         radio_sel;
    char        radio_opts[8][32];
    int         radio_count;
    int         num_val;
    int         num_min, num_max;
    float       float_val;
    struct Menu *submenu;
} ConfigItem;

/* ── menu ───────────────────────────────────────────────────────── */
typedef struct Menu {
    char       title[MAX_LABEL];
    ConfigItem items[MAX_ITEMS];
    int        count;
    int        cursor;
    int        scroll;
} Menu;

/* ── NeuralcConfig ──────────────────── */
typedef struct {
    /* Performance */
    int   use_omp;
    int   omp_auto;
    int   omp_threads;
    int   use_gpu;
    int   use_blas;

    /* Training & Hyperparameters */
    int   hidden_size;
    int   seq_len;
    float temperature;
    int   batch_size;
    float learning_rate;
    int   optimizer;       /* 0=Adam 1=SGD 2=RMSProp */
    float dropout_rate;
    int   epochs;
    int   use_grad_clip;
    float grad_clip;

    /* Memory */
    int   allocator;
    int   pool_size_mb;

    /* Debug */
    int   debug_mode;
    int   check_nan;
    int   profile;

    /* Model Engine Selection */
    int   active_model_type; /* 0=RNN, 1=Transformer */

    /* Transformer Engine Hyperparameters */
    int   trans_vocab_size;
    int   trans_dim;
    int   trans_hidden_dim;
    int   trans_n_layers;
    int   trans_n_heads;
    int   trans_max_seq_len;

    /* Build */
    int   opt_level;
    int   enable_avx;
    int   enable_lto;
} NeuralcConfig;

/* ── API ────────────────────────────────────────────────────────── */

/* Hardware detection  */
void detect_system_hardware(HardwareProfile *prof);

/* Config lifecycle */
void config_defaults(NeuralcConfig *cfg);
int  config_load(NeuralcConfig *cfg, const char *path);
int  config_save(const NeuralcConfig *cfg, const char *path);
void config_print(const NeuralcConfig *cfg);

/* Run interactive UI — returns 0 if saved, 1 if quit */
int  config_ui_run(NeuralcConfig *cfg);

#endif /* CONFIG_UI_H */
