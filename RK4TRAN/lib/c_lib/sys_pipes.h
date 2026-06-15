#ifndef SYS_PIPES_H
#define SYS_PIPES_H

#include <stdio.h>

void* c_open_gnuplot_pipe(const char* command);
void c_write_gnuplot_pipe(void* pipe_ptr, const char* str);
void c_close_gnuplot_pipe(void* pipe_ptr);

#endif
