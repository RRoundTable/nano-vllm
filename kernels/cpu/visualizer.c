#include "ops.h"
#include <stdio.h>
#include <math.h>

// ANSI Color Codes
#define C_RESET  "\033[0m"
#define C_RED    "\033[31m"
#define C_GREEN  "\033[32m"
#define C_YELLOW "\033[33m"
#define C_BLUE   "\033[34m"
#define C_MAGENTA "\033[35m"
#define C_CYAN   "\033[36m"
#define C_GRAY   "\033[90m"

// Helper to print a progress bar or heatmap intensity
void print_intensity(float val) {
    // Simple heatmap: 
    // < 0.1 : .
    // < 0.3 : -
    // < 0.5 : +
    // < 0.7 : *
    // < 0.9 : #
    // >= 0.9: @
    
    if (val < 0.1f) printf(C_GRAY "." C_RESET);
    else if (val < 0.3f) printf(C_BLUE "-" C_RESET);
    else if (val < 0.5f) printf(C_CYAN "+" C_RESET);
    else if (val < 0.7f) printf(C_GREEN "*" C_RESET);
    else if (val < 0.9f) printf(C_YELLOW "#" C_RESET);
    else printf(C_RED "@" C_RESET);
}

void visualize_attention(float* att, int n_heads, int pos, int max_seq_len) {
    // Visualize only the first 4 heads to save space
    int heads_to_show = n_heads > 4 ? 4 : n_heads;
    
    printf("\n[Visualizer] Attention Patterns (Step %d)\n", pos);
    printf("Head: ");
    for (int h = 0; h < heads_to_show; h++) printf("  H%02d   ", h);
    printf("\n      ");
    for (int h = 0; h < heads_to_show; h++) printf("------- ");
    printf("\n");
    
    // Show last 10 tokens context or full context if small
    int start_pos = pos > 10 ? pos - 10 : 0;
    
    for (int t = start_pos; t <= pos; t++) {
        printf("t%03d: ", t);
        for (int h = 0; h < heads_to_show; h++) {
            float val = att[h * max_seq_len + t];
            // Print bar chart
            printf("[");
            print_intensity(val);
            print_intensity(val * 2);
            print_intensity(val * 4);
            printf("] ");
        }
        printf("\n");
    }
    printf("\n");
}

void visualize_kv_cache_usage(int layer, int pos, int max_seq_len) {
    if (layer != 0) return; // Only show for layer 0 to avoid spam
    
    float usage = (float)(pos + 1) / max_seq_len * 100.0f;
    printf("[Visualizer] KV Cache Usage: %d/%d tokens (%.1f%%)\n", pos+1, max_seq_len, usage);
    
    // Draw memory bar
    int bar_width = 50;
    int filled = (int)(usage / 100.0f * bar_width);
    printf("Mem: [");
    for (int i=0; i<bar_width; i++) {
        if (i < filled) printf(C_GREEN "#" C_RESET);
        else printf(C_GRAY "." C_RESET);
    }
    printf("]\n");
}

