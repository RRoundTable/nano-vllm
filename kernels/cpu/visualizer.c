#include "ops.h"
#include <stdio.h>
#include <math.h>
#include "log.h"

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
    
    if (val < 0.1f) log_printf(C_GRAY "." C_RESET);
    else if (val < 0.3f) log_printf(C_BLUE "-" C_RESET);
    else if (val < 0.5f) log_printf(C_CYAN "+" C_RESET);
    else if (val < 0.7f) log_printf(C_GREEN "*" C_RESET);
    else if (val < 0.9f) log_printf(C_YELLOW "#" C_RESET);
    else log_printf(C_RED "@" C_RESET);
}

void visualize_attention(float* att, int n_heads, int pos, int max_seq_len) {
    // Visualize only the first 4 heads to save space
    int heads_to_show = n_heads > 4 ? 4 : n_heads;
    
    log_printf("\n[Visualizer] Attention Patterns (Step %d)\n", pos);
    log_printf("Head: ");
    for (int h = 0; h < heads_to_show; h++) log_printf("  H%02d   ", h);
    log_printf("\n      ");
    for (int h = 0; h < heads_to_show; h++) log_printf("------- ");
    log_printf("\n");
    
    // Show last 10 tokens context or full context if small
    int start_pos = pos > 10 ? pos - 10 : 0;
    
    for (int t = start_pos; t <= pos; t++) {
        log_printf("t%03d: ", t);
        for (int h = 0; h < heads_to_show; h++) {
            float val = att[h * max_seq_len + t];
            // Print bar chart
            log_printf("[");
            print_intensity(val);
            print_intensity(val * 2);
            print_intensity(val * 4);
            log_printf("] ");
        }
        log_printf("\n");
    }
    log_printf("\n");
}

// Mode: 0 = Linear (Naive) Fragmentation Demo, 1 = Paged (Block-based) Fragmentation Demo
void visualize_kv_cache_usage(int layer, int pos, int max_seq_len, int mode) {
    if (layer != 0) return; // Only show for layer 0 to avoid spam
    
    // Fragmentation Demo: Simulate a memory arena with multiple requests
    // Let's assume we have 4 requests running concurrently to show the effect clearly.
    // Req 0: The actual current request (pos + 1 tokens)
    // Req 1: A finished short request (e.g., 50 tokens)
    // Req 2: A long request (e.g., 200 tokens)
    // Req 3: A new request (e.g., 10 tokens)
    
    int req_lens[4] = { pos + 1, 50, 200, 10 };
    int num_reqs = 4;
    
    log_printf("\n[Visualizer] Memory Breakdown (Mode: %s)\n", mode == 1 ? "Paged" : "Naive");
    
    int total_used = 0;
    for(int i=0; i<num_reqs; i++) total_used += req_lens[i];

    if (mode == 1) {
        // Paged View: 
        // - Used: Green
        // - Reserved: None (0)
        // - Internal Frag: Yellow (Leftover in last block)
        int block_size = 16;
        int total_alloc = 0;
        int internal_frag = 0;
        
        for(int i=0; i<num_reqs; i++) {
            int blocks = (req_lens[i] + block_size - 1) / block_size;
            int alloc = blocks * block_size;
            total_alloc += alloc;
            internal_frag += (alloc - req_lens[i]);
        }
        
        log_printf("Reqs: ");
        for(int i=0; i<num_reqs; i++) {
            int blocks = (req_lens[i] + block_size - 1) / block_size;
            log_printf("[Req%d:%d Blks] ", i, blocks);
        }
        log_printf("\n\n");

        // Visualization Bar
        int total_slots = total_alloc; // Just show allocated space
        int bar_width = 50;
        int used_chars = (int)((float)total_used / total_slots * bar_width);
        int frag_chars = bar_width - used_chars;
        
        log_printf("Memory Map:\n[");
        for(int i=0; i<used_chars; i++) log_printf(C_GREEN "#" C_RESET);
        for(int i=0; i<frag_chars; i++) log_printf(C_YELLOW "." C_RESET);
        log_printf("]\n");
        
        log_printf("1. Used (Actual Data)        : %d (%.1f%%)\n", total_used, (float)total_used/total_alloc*100);
        log_printf("2. Reserved (Future Tokens)  : 0 (0.0%%)\n");
        log_printf("3. Internal Frag (Last Page) : " C_YELLOW "%d (%.1f%%)" C_RESET "\n", internal_frag, (float)internal_frag/total_alloc*100);
        
    } else {
        // Naive View:
        // - Used: Green
        // - Reserved: Red (Huge waste)
        // - Internal Frag: None (Included in Reserved logic for Contiguous)
        int total_alloc = num_reqs * max_seq_len;
        int reserved_waste = total_alloc - total_used;
        
        log_printf("Reqs: ");
        for(int i=0; i<num_reqs; i++) log_printf("[Req%d:%d/%d] ", i, req_lens[i], max_seq_len);
        log_printf("\n\n");

        // Visualization Bar
        int bar_width = 50;
        int used_chars = (int)((float)total_used / total_alloc * bar_width);
        int reserved_chars = bar_width - used_chars;
        if (used_chars == 0 && total_used > 0) { used_chars = 1; reserved_chars--; }
        
        log_printf("Memory Map:\n[");
        for(int i=0; i<used_chars; i++) log_printf(C_GREEN "#" C_RESET);
        for(int i=0; i<reserved_chars; i++) log_printf(C_RED "." C_RESET);
        log_printf("]\n");
        
        log_printf("1. Used (Actual Data)        : %d (%.1f%%)\n", total_used, (float)total_used/total_alloc*100);
        log_printf("2. Reserved (Future Tokens)  : " C_RED "%d (%.1f%%)" C_RESET " -> CRITICAL WASTE\n", reserved_waste, (float)reserved_waste/total_alloc*100);
        log_printf("3. Internal Frag (Last Page) : 0 (N/A for Naive)\n");
    }
    log_printf("\n");
}
