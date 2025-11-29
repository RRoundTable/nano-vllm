#include "log.h"
#include <stdlib.h>
#include <string.h>

static FILE* log_file = NULL;

void log_init(const char* filename) {
    if (filename) {
        log_file = fopen(filename, "w");
        if (!log_file) {
            fprintf(stderr, "Failed to open log file: %s\n", filename);
            // Non-fatal, continue without file logging
        }
    }
}

void log_close() {
    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
}

// Helper to strip ANSI codes from string
// e.g. "\033[31mHello\033[0m" -> "Hello"
static void strip_ansi_codes(char* dest, const char* src) {
    while (*src) {
        if (*src == '\033' && *(src+1) == '[') {
            // Found escape sequence start
            src += 2; // Skip \033[
            // Skip until 'm' or other terminator, or end of string
            // Simple ANSI color codes end with 'm'
            // But to be safe, we skip numeric parameters and ;
            while (*src && ((*src >= '0' && *src <= '9') || *src == ';')) {
                src++;
            }
            if (*src == 'm') {
                src++; // Skip terminator
            }
        } else {
            *dest++ = *src++;
        }
    }
    *dest = '\0';
}

void log_printf(const char* fmt, ...) {
    va_list args;
    
    // 1. Print to stdout (with colors)
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    // 2. Print to file (without colors)
    if (log_file) {
        // Format the full string first
        char buffer[4096]; // Assuming log lines aren't excessively long
        
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        
        // Strip ANSI codes
        char clean_buffer[4096];
        strip_ansi_codes(clean_buffer, buffer);
        
        // Write to file
        fprintf(log_file, "%s", clean_buffer);
        fflush(log_file);
    }
}
