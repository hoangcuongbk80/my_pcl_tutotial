
#define cl_rows 480
#define cl_cols 640

__kernel void GPU_NormalCompute(__global float* depth, __global float3* normals)
    {
       int row = get_global_id(0); int col = get_global_id(1);
       int index = cl_cols*row + col;
       if(depth[index] == 0 || row == 0 || row == cl_rows-1 || col == 0 || col == cl_cols-1) 
           {
              normals[index].xyz = (float3)(0.0f, 0.0f, 0.0f);
           }
        else
        {
              normals[index].x = (depth[index + cl_cols] - depth[index - cl_cols]) / 2.0; //depth(row+1, col)-depth(row-1, col)
              normals[index].y = (depth[index + 1] - depth[index - 1]) / 2.0; //depth(row, col+1)-depth)row, col-1)
        }
    }