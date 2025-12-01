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

// Mode: 0 = Linear (Naive), 1 = Paged (Block-based)
void visualize_kv_cache_usage(Sequence* seqs, int num_seqs, KVCacheManager* mgr, int max_seq_len, int mode) {
    // Calculate real stats
    int total_used = 0;
    for (int i = 0; i < num_seqs; i++) {
        total_used += (seqs[i].pos + 1);
    }

    log_printf("\n[Visualizer] Memory Breakdown (Mode: %s)\n", mode == 1 ? "Paged" : "Naive");

    if (mode == 1) {
        // Paged View
        int block_size = mgr->block_size;
        int total_blocks = mgr->num_blocks;
        int total_alloc = total_blocks * block_size; // Total physical capacity
        
        int used_blocks = 0;
        for (int i = 0; i < num_seqs; i++) {
            used_blocks += seqs[i].table.num_blocks;
        }
        int allocated_mem = used_blocks * block_size;
        int internal_frag = allocated_mem - total_used;
        int free_mem = total_alloc - allocated_mem;
        
        log_printf("Reqs: ");
        for(int i=0; i<num_seqs; i++) {
            log_printf("[Req%d:%d Blks] ", i, seqs[i].table.num_blocks);
        }
        log_printf("\n\n");

        // Visualization Bar
        int bar_width = 50;
        // Normalized to Total Physical Pool Size
        float scale = (float)bar_width / total_alloc;
        
        int used_chars = (int)(total_used * scale);
        int frag_chars = (int)(internal_frag * scale);
        int free_chars = (int)(free_mem * scale);
        
        // Adjust rounding
        if (used_chars + frag_chars + free_chars < bar_width) free_chars = bar_width - (used_chars + frag_chars);

        log_printf("Memory Map (Total Pool: %d blocks = %d slots):\n[", total_blocks, total_alloc);
        for(int i=0; i<used_chars; i++) log_printf(C_GREEN "#" C_RESET);
        for(int i=0; i<frag_chars; i++) log_printf(C_YELLOW "." C_RESET);
        for(int i=0; i<free_chars; i++) log_printf(C_GRAY "_" C_RESET);
        log_printf("]\n");
        
        log_printf("1. Used (Actual Data)        : %d (%.1f%%)\n", total_used, (float)total_used/total_alloc*100);
        log_printf("2. Internal Frag (Last Page) : " C_YELLOW "%d (%.1f%%)" C_RESET "\n", internal_frag, (float)internal_frag/total_alloc*100);
        log_printf("3. Free (Available Blocks)   : %d (%.1f%%)\n", free_mem, (float)free_mem/total_alloc*100);

    } else {
        // Naive View
        // In naive mode, each sequence grabs 'max_seq_len' immediately.
        // We simulate this simply by multiplying num_seqs * max_seq_len
        int total_alloc = num_seqs * max_seq_len; 
        int reserved_waste = total_alloc - total_used;
        
        log_printf("Reqs: ");
        for(int i=0; i<num_seqs; i++) log_printf("[Req%d:%d/%d] ", i, seqs[i].pos + 1, max_seq_len);
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
