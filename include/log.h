#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdarg.h>

// Initialize logger (opens file)
void log_init(const char* filename);

// Close logger
void log_close();

// Print to both stdout and file
void log_printf(const char* fmt, ...);

#endif // LOG_H

