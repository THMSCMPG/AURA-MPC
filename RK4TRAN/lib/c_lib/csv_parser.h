#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <stddef.h>

// Struct to pass dynamically allocated CSV data back to Fortran
typedef struct {
    char var_name[64];
    double* values;
    int num_elements;
} InitialConditionC;

// Exported functions
void read_independent_csv(const char* filepath, InitialConditionC* ic_data);
void free_ic_data(InitialConditionC* ic_data);
void get_next_prediction_filename(char* filename, size_t max_len);

#endif
