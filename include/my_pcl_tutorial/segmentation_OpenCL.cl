
#define cl_rows 480
#define cl_cols 640
#define gausize 7
#define bilateral 0.05
#define depthRE 0.001
#define noiseThresh 0.005

__kernel void Pre_Processing(__global float* gaukernel, __global float* depth, __global float* pdepth)
    {
       int row = get_global_id(0); int col = get_global_id(1);
       int index = cl_cols*row + col;
       pdepth[index] = 0;
       for(int subrow = 0; subrow < gausize; subrow++) 
       {
            for(int subcol = 0; subcol < gausize ; subcol++) 
            {
                int neighborRow = row + subrow - gausize / 2;
                int neighborCol = col + subcol - gausize / 2;
                if(neighborRow > cl_rows || neighborCol > cl_cols || neighborRow < 0 || neighborCol < 0)
                {
                    pdepth[index] += gaukernel[subrow * gausize + subcol] * depth[index]; continue;
                }
                int neighborindex = cl_cols * neighborRow + neighborCol;
                if(fabs(depth[index] - depth[neighborindex]) > bilateral)
                {
                    pdepth[index] += gaukernel[subrow * gausize + subcol] * depth[index]; continue;
                }
                pdepth[index] += gaukernel[subrow * gausize + subcol] * depth[neighborindex];
            }
        }
    }

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

              // Remove inaccurate point based on angle of normal (x, y, depthRE) to camera (0, 0, depthRE)
              if(fabs(normals[index].x) + fabs(normals[index].y) > noiseThresh) normals[index].xyz = (float3)(0.0f, 0.0f, 0.0f);
        }
    }