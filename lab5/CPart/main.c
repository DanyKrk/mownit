#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <gsl/gsl_blas.h>


clock_t st_time, en_time;
struct tms st_cpu, en_cpu;
char* report_file_name = "c_data.csv";


void save_timer(int matrix_size, char *operation_type, FILE *f){
    int clk_tics = sysconf(_SC_CLK_TCK);
    double real_time = (double)(en_time - st_time) / clk_tics;
    printf("%d,%s,%f\n",
           matrix_size,
           operation_type,
           real_time);
    fprintf(f,"%d,%s,%f\n",
           matrix_size,
           operation_type,
           real_time);
}

void start_timer(){
    st_time = times(&st_cpu);
}

void stop_timer(){
    en_time = times(&en_cpu);
}


void write_report_header(FILE *f){
    fprintf(f, "Matrix_size,Operation_type,Operation_time\n");
    printf( "Matrix_size,Operation_type,Operation_time\n");
}

void naive_multiply(double **A, double **B, double **C, int n){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k]*B[k][j];
}

void better_multiply(double **A, double **B, double **C, int n){
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k]*B[k][j];
}

double **create_random_matrix(int n){
    double **A = calloc(n, sizeof(double *));
    for (int i = 0; i < n; ++i) {
        A[i] = calloc(n, sizeof(double ));
        for (int j = 0; j < n; ++j) {
            A[i][j] = ((double)rand()/(double)(RAND_MAX)) * 5.0;
        }
    }
    return A;
}
double **create_zeroed_matrix(int n){
    double **A = calloc(n, sizeof(double *));
    for (int i = 0; i < n; ++i) {
        A[i] = calloc(n, sizeof(double ));
        for (int j = 0; j < n; ++j) {
            A[i][j] = 0.0;
        }
    }
    return A;
}
void free_matrix(double **A, int n){
    for (int i = 0; i < n; ++i) {
        free(A[i]);
    }
    free(A);
}

int main(){
    FILE* report_file = fopen(report_file_name, "w");
    write_report_header(report_file);
    int strt = 10;
    int end = 1000;
    int stp = 75;

    for(int i = strt; i <= end; i+=stp){
        for(int try_id = 1; try_id<= 10; try_id++){
            int curr_matrix_size = i;
            double **A, **B, **C;
            A = create_random_matrix(curr_matrix_size);
            B = create_random_matrix(curr_matrix_size);
            C = create_zeroed_matrix(curr_matrix_size);

            start_timer();
            naive_multiply(A,B,C,curr_matrix_size);
            stop_timer();
            save_timer(curr_matrix_size, "naive_multiplication", report_file);
        }
    }
    for(int i = strt; i <= end; i+=stp){
        for(int try_id = 1; try_id<= 10; try_id++){
            int curr_matrix_size = i;
            double **A, **B, **C;
            A = create_random_matrix(curr_matrix_size);
            B = create_random_matrix(curr_matrix_size);
            C = create_zeroed_matrix(curr_matrix_size);

            start_timer();
            better_multiply(A,B,C,curr_matrix_size);
            stop_timer();
            save_timer(curr_matrix_size, "better_multiplication", report_file);
        }
    }

    for(int i = strt; i <= end; i+=stp){
        for(int try_id = 1; try_id<= 10; try_id++){
            int curr_matrix_size = i;
            double *A = calloc(curr_matrix_size * curr_matrix_size, sizeof(double));
            double *B = calloc(curr_matrix_size * curr_matrix_size, sizeof(double));
            double *C = calloc(curr_matrix_size * curr_matrix_size, sizeof(double));
            for (int k = 0; k < curr_matrix_size * curr_matrix_size; k++) {
                A[k] = ((double)rand()/(double)(RAND_MAX)) * 5.0;
                B[k] = ((double)rand()/(double)(RAND_MAX)) * 5.0;
                C[k] = 0.0;
            }
            gsl_matrix_view A_GSL = gsl_matrix_view_array(A,i,i);
            gsl_matrix_view B_GSL = gsl_matrix_view_array(B,i,i);
            gsl_matrix_view C_GSL = gsl_matrix_view_array(C,i,i);
            start_timer();
            gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                            1.0, &A_GSL.matrix, &B_GSL.matrix,
                            0.0, &C_GSL.matrix);
            stop_timer();
            save_timer(curr_matrix_size, "BLAS_multiplication", report_file);
            free(A);
            free(B);
            free(C);
        }
    }
}