#define MAT_SIZE 100
#define MAT_NUMEL 10000
#define TOTAL_NUMEL 2560000
#define NUM_MATR 256
#define NUM_LOOP 200

__kernel void GPU_First_Task(__global unsigned char* offset, __global unsigned char* X, 
                             __global unsigned char* Z, __global unsigned char* W)
    {
       int index = get_global_id(0);
       int depth = index / (MAT_NUMEL);
       int X_index = index % MAT_NUMEL;
       W[index] = Z[index] + X[X_index] * offset[depth]; 
    }

__kernel void GPU_Second_Task(__global unsigned char* offset, __global unsigned char* Y, 
                             __global unsigned char* Z, __global unsigned char* W)
    {
       int index = get_global_id(0);
       int depth = index / (MAT_NUMEL);
       W[index] = Z[index] + offset[depth]; 
    }