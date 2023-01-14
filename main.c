#include<stdio.h>
#include<stdlib.h>
#define __USE_C99_MATH
#include<string.h>
#include<stdbool.h>
#include<math.h>
#include<mpi.h>
#include <omp.h>
#include<sys/time.h>

bool test_case1();

long int product(int *array, int n) {
    long int product = 1;
    for(int i=0; i<n; i++) {
        product *= array[i];
    }
    return product;
}

int *read_dims(char *filename) {
    FILE *file = fopen(filename,"r");
    
    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return NULL;
    }

    char firstline[500];
    fgets(firstline, 500, file);
    
    int line_length = strlen(firstline);

    int num_dims = 0;
    for(int i=0; i<line_length; i++) {
        if(firstline[i] == ' ') {
            num_dims++;
        }
    }
    
    int *dims = malloc((num_dims+1)*sizeof(int));
    dims[0] = num_dims;
    const char s[2] = " ";
    char *token;
    token = strtok(firstline, s);
    int i = 0;
    while( token != NULL ) {
        dims[i+1] = atoi(token);
        i++;
        token = strtok(NULL, s);
    }
    fclose(file);
    return dims;
}

float * read_array(char *filename, int *dims, int num_dims) {
    FILE *file = fopen(filename,"r");

    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return NULL;
    }

    char firstline[500];
    fgets(firstline, 500, file);

    //Ignore first line and move on since first line contains 
    //header information and we already have that. 

    long int total_elements = product(dims, num_dims);

    float *one_d = malloc(sizeof(float) * total_elements);
    for(int i=0; i<total_elements; i++) {
        fscanf(file, "%f", &one_d[i]);
    }
    fclose(file);
    return one_d;
}

int write_array(char *filename, int m, int p, float *output){
    int size = m * p;
    FILE *file = fopen(filename,"w");

    if(file == NULL) {
        printf("Unable to open file: %s", filename);
        return -1;
    }

    if (file != NULL) {
        fprintf(file, "%d ", m);
        fprintf(file, "%d ", p);;
        fprintf(file, "\n");
    }

    for(int i=0; i<size; i++) {
        fprintf(file, "%.6f ", output[i]);
    }

    fclose(file);
    return 1;
}

int main(int argc, char *argv[]) {
    //Define the number and id of threads
    int process_id, process_num;
    //Packets for broadcasting
    int matrix_properties[6];
    //Time Record
    double MPI_timer[4];
    //Raw data read from thread 0, belongs to address header, no space allocated
    float *a_data = NULL;
    float *b_data = NULL;
    float *cal_recv = NULL;
    //The final output to the space, which is now still the address header
    float *c = NULL;
    bool match = true;
    char a_filename[500];
    char b_filename[500];
    char check_filename[500];
    MPI_Init(&argc, &argv);
    MPI_timer[0] = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_num);
    if(process_id == 0){
        if(argc != 4) {
            printf("Usage: %s <filename_input> <filename_kernel> <filename_expected_output>\n", argv[0]);
            return -1;
        }
        int compareOutput = 1;

        strcpy(a_filename, argv[1]);
        strcpy(b_filename, argv[2]);
        strcpy(check_filename, argv[3]);

        int *a_dims_original = read_dims(a_filename);
        if(a_dims_original == NULL) {
            return -1;
        }

        int a_num_dims = a_dims_original[0];
        int *a_dims = a_dims_original+1;
        a_data = read_array(a_filename, a_dims, a_num_dims);
        if(a_data == NULL) {
            return -1;
        }
        int *b_dims_original = read_dims(b_filename);
        if(b_dims_original == NULL) {
            return -1;
        }
        int b_num_dims = b_dims_original[0];
        int *b_dims = b_dims_original+1;
        b_data = read_array(b_filename, b_dims, b_num_dims);
        if(b_data == NULL) {
            return -1;
        }
        //Place all parameters to be used in the broadcast package
        matrix_properties[0] = a_dims[0];//m
        matrix_properties[1] = b_dims[0];//n
        matrix_properties[2] = b_dims[1];//p
        int size_cal = a_dims[0] * b_dims[1];
        int max_np;
        if(size_cal < process_num){
            max_np = size_cal;
        }else{
            max_np = process_num;
        }
        printf("calculation unit %d, have threads %d, working threads %d\n", size_cal, process_num, max_np);
        int elements = (int)(size_cal / max_np);
        int rest = size_cal - (elements * max_np);
        printf("each thread get %d, the last thread get %d\n", elements, rest + elements);
        matrix_properties[3] = max_np;
        matrix_properties[4] = elements;
        matrix_properties[5] = rest;
    }
    MPI_timer[1] = MPI_Wtime();
    //Broadcast parameters to all threads
    MPI_Bcast(&matrix_properties, 6, MPI_INT, 0, MPI_COMM_WORLD);
    int size_a = matrix_properties[0] * matrix_properties[1];//m*n
    int size_b = matrix_properties[1] * matrix_properties[2];//n*p
    int size_c = matrix_properties[0] * matrix_properties[2];//m*p
    //Create space for data to be placed after each thread's calculation
    int array_size = matrix_properties[4] + matrix_properties[5];//elements + rest
    //Create memory space for a and b storage for each process
    float *a = malloc(sizeof(float) * size_a);
    float *b = malloc(sizeof(float) * size_b);
    //Definition of packets used by gather
    float *cal_back = malloc(sizeof(float) * array_size);
    if(process_id == 0){
        //Initialize in process 0, input and kernel in all processes
        a = a_data;
        b = b_data;
        //Creat memory for output in process 0, when only process 0 can use output
        c = malloc(sizeof(float) * size_c);
        cal_recv = malloc(sizeof(float) * (process_num * array_size));
    }
    //Broadcast input and kernel to all processes
    MPI_Bcast(a, size_a, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, size_b, MPI_FLOAT, 0, MPI_COMM_WORLD);

    //coordinate[0] = start_row, coordinate[1] = start_col, coordinate[2] = elements
    //coordinate[3] = flag, if this threads have task flag = 1, if not flag = 0;
    int *coordinate = malloc(sizeof(int) * 4);
    if(process_id > matrix_properties[4]){
        *(coordinate + 0) = 0;
        *(coordinate + 1) = 0;
        *(coordinate + 2) = 0;
        *(coordinate + 3) = 0;
        //MPI_Send(coordinate, 4, MPI_INT, process_id, 99, MPI_COMM_WORLD);
    }else{
        //Calculate the start address of each thread and the number of data to be calculated
        int elements_cal;
        if(process_id < matrix_properties[3] - 1){
            elements_cal = matrix_properties[4];
        }else{
            elements_cal = matrix_properties[4] + matrix_properties[5];
        }
        int already_sent = process_id * matrix_properties[4];
        int start_row = (int)(already_sent / matrix_properties[2]);
        int start_col = already_sent - (start_row * matrix_properties[2]);
        *(coordinate + 0) = start_row;
        *(coordinate + 1) = start_col;
        *(coordinate + 2) = elements_cal;
        *(coordinate + 3) = 1;
    }

    //Threads start computing
    int row_recv = *(coordinate + 0);
    int col_recv = *(coordinate + 1);
    int elements = *(coordinate + 2);
    int flag = *(coordinate + 3);
    if(flag == 1){
        //Effective threads for work
        for(int e = 0; e < elements; e++){
            //Arithmetic from the start address
            int relocate = row_recv * matrix_properties[2] + col_recv + e;
            int new_row = (int)(relocate / matrix_properties[2]);
            int new_col = relocate - (new_row * matrix_properties[2]);
            float temp = 0;
            for(int position = 0; position < matrix_properties[1]; position++){
                temp += a[new_row * matrix_properties[1] + position] * b[new_col + position * matrix_properties[2]];
            }
            *(cal_back + e) = temp;
        }
    }else{
        //Non-valid threads for work
        printf("Rank %d is not working \n", process_id);
    }
    //gather the results of each thread's operations
    MPI_Gather(cal_back, array_size, MPI_FLOAT, cal_recv, array_size,  MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(process_id == 0){
        //Final redistribution of data at process 0
        int counting = 0;
        for (int t = 0; t < matrix_properties[3] - 1; t++)
        {
            for(int e = 0; e < elements; e++){
                *(c + counting) = *(cal_recv + t * array_size + e);
                counting++;
            }
        }
        for(int er = 0; er < array_size; er++){
            *(c + counting) = *(cal_recv + (matrix_properties[3] - 1) * array_size + er);
            counting++;
        }
        MPI_timer[2] = MPI_Wtime();
        printf("%f\n", MPI_timer[2] - MPI_timer[1]);
        //Process 0 does the file writing
        int write = write_array(check_filename, matrix_properties[0], matrix_properties[2], c);
        if(write == 1) {
            printf("Writing successful!\n");
        }
    }
    MPI_Finalize();
    return !match;
}
