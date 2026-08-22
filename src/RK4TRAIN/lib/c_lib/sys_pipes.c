#include "sys_pipes.h"
#include <stdlib.h>

void* c_open_gnuplot_pipe(const char* command) {
	FILE* pipe = popen(command, "w");
	return (void*)pipe;
}

void c_write_gnuplot_pipe(void* pipe_ptr, const char* str) {
	FILE* pipe = (FILE*)pipe_ptr;
	if (pipe != NULL) {
		fprintf(pipe, "%s", str);
		fflush(pipe);
	}
}

void c_close_gnuplot_pipe(void* pipe_ptr) {
	FILE* pipe = (FILE*)pipe_ptr;
	if (pipe != NULL) {
		pclose(pipe);
	}
}
