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

void visualize_scheduler_status(Sequence* seqs, int num_seqs) {
    log_printf("\n[Scheduler Status]\n");
    for (int i = 0; i < num_seqs; i++) {
        log_printf("Seq %d: ", i);
        
        switch (seqs[i].status) {
            case SEQ_WAITING:
                log_printf(C_GRAY "[WAITING]" C_RESET " (Arrives at Step %d)", seqs[i].arrival_step);
                break;
            case SEQ_PREFILLING: {
                int total = seqs[i].num_prompt_tokens;
                int current = seqs[i].pos;
                float progress = (float)current / total;
                int bars = (int)(progress * 20);
                
                log_printf(C_YELLOW "[PREFILL]" C_RESET " (%3d/%3d) [", current, total);
                for(int b=0; b<bars; b++) log_printf("=");
                for(int b=bars; b<20; b++) log_printf(" ");
                log_printf("]");
                break;
            }
            case SEQ_DECODING: {
                int gen = seqs[i].pos - seqs[i].num_prompt_tokens;
                log_printf(C_GREEN "[DECODE ]" C_RESET " (Gen: %3d) " C_GREEN "Generating..." C_RESET, gen);
                break;
            }
            case SEQ_FINISHED:
                log_printf(C_BLUE "[DONE   ]" C_RESET " Finished.");
                break;
        }
        log_printf("\n");
    }
    log_printf("\n");
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
    
    // 1. Show Scheduler Status first
    visualize_scheduler_status(seqs, num_seqs);

    // Calculate real stats
    int total_used = 0;
    for (int i = 0; i < num_seqs; i++) {
        total_used += (seqs[i].pos + 1);
    }

    log_printf("[Visualizer] Memory Breakdown (Mode: %s)\n", mode == 1 ? "Paged" : "Naive");

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

void visualize_final_timeline(char* history, int num_seqs, int total_steps, int max_steps_capacity) {
    log_printf("\n=== Execution Timeline ===\n");
    
    // Print Header: 01 02 03 ...
    // To save space, let's just print blocks and use tooltips logic in mind.
    // Or simplified header
    log_printf("Step: ");
    for (int t = 0; t < total_steps; t++) {
        if (t % 10 == 0) log_printf("%-2d", t);
        else log_printf("  "); 
    }
    log_printf("\n");

    for (int i = 0; i < num_seqs; i++) {
        log_printf("Seq%d: ", i);
        for (int t = 0; t < total_steps; t++) {
            char status = history[i * max_steps_capacity + t];
            switch (status) {
                case 'P': log_printf(C_YELLOW "P " C_RESET); break;
                case 'D': log_printf(C_GREEN "D " C_RESET); break;
                case 'W': log_printf(C_GRAY ". " C_RESET); break;
                case 'F': log_printf(C_BLUE "F " C_RESET); break;
                default:  log_printf("  "); break;
            }
        }
        log_printf("\n");
    }
    log_printf("Legend: " C_GRAY "." C_RESET "=Wait, " C_YELLOW "P" C_RESET "=Prefill, " C_GREEN "D" C_RESET "=Decode, " C_BLUE "F" C_RESET "=Finished\n");
    log_printf("==========================\n\n");
}
