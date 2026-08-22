#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>

typedef struct {
    char var_name[64];
    double* values;
    int num_elements;
} InitialConditionC;

int get_directory_file_count(const char* dir_path) {
    DIR *d = opendir(dir_path);
    if (!d) return 0;
    
    struct dirent *dir;
    int count = 0;
    while ((dir = readdir(d)) != NULL) {
        if (strstr(dir->d_name, ".csv") != NULL) {
            count++;
        }
    }
    closedir(d);
    return count;
}

void get_directory_filepaths(const char* dir_path, char paths[][256], int start_idx, int max_files) {
    DIR *d = opendir(dir_path);
    if (!d) return;
    
    struct dirent *dir;
    int i = start_idx;
    while ((dir = readdir(d)) != NULL && i < max_files) {
        if (strstr(dir->d_name, ".csv") != NULL) {
            snprintf(paths[i], 256, "%s/%s", dir_path, dir->d_name);
            i++;
        }
    }
    closedir(d);
}

void read_independent_csv(const char* filepath, InitialConditionC* ic_data) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Error opening CSV file");
        ic_data->num_elements = 0;
        ic_data->values = NULL;
        return;
    }

    char line[1024];
    if (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\r\n")] = 0;
        strncpy(ic_data->var_name, line, 63);
        ic_data->var_name[63] = '\0';
    }

    int rows = 0;
    long data_pos = ftell(file);
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) rows++;
    }

    ic_data->num_elements = rows;
    ic_data->values = (double*)malloc(rows * sizeof(double));

    fseek(file, data_pos, SEEK_SET);
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < rows) {
        ic_data->values[i] = strtod(line, NULL);
        i++;
    }

    fclose(file);
}

void free_ic_data(InitialConditionC* ic_data) {
    if (ic_data->values) {
        free(ic_data->values);
        ic_data->values = NULL;
    }
}

void get_next_prediction_filename(char* filename, size_t max_len) {
    int run_idx = 1;
    do {
        snprintf(filename, max_len, "independent/live_prediction%04d.csv", run_idx);
        if (access(filename, F_OK) != 0) {
            return;
        }
        run_idx++;
    } while (run_idx <= 9999);
}
