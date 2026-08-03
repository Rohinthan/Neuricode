/*
 * neuralc_config_main.c — entry point for neuralc config UI
 *
 * Run:  make config  (auto-runs menuconfig)
 *
 * Loads existing config (if any), runs the interactive UI,
 * saves neuralc_config.h on exit.
 */

#include "config_ui.h"
#include <stdio.h>
#include <stdlib.h>

#define CONFIG_PATH "neuralc_config.h"

int main(void) {
    NeuralcConfig cfg;

    /* load existing config or use defaults */
    if (config_load(&cfg, CONFIG_PATH) == 0)
        printf("Loaded existing config from %s\n", CONFIG_PATH);
    else
        config_defaults(&cfg);

    /* run interactive UI */
    int result = config_ui_run(&cfg);

    if (result == 0) {
        /* saved */
        if (config_save(&cfg, CONFIG_PATH) == 0) {
            printf("\n✓ Configuration saved to %s\n\n", CONFIG_PATH);
            config_print(&cfg);
            printf("Now rebuild with:\n");
            printf("  make clean && make\n\n");
        } else {
            fprintf(stderr, "Error: could not save config!\n");
            return 1;
        }
    } else {
        printf("\nConfiguration not saved.\n");
    }

    return 0;
}
